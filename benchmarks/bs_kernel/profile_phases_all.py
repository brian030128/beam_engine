"""Cross-method per-phase timing breakdown.

Runs beam search on a fixed (K, L_p, B, max_new) cell for paged / mlca /
fasttree / bs_kernel and reports mean / median / total of every decode
phase (cow, plan, forward, topk, fork) per method. For bs_kernel it also
dumps the cost-model picks (strategy + depth + pool_count + t_large)
histogram across decode steps.

This is the cross-method analog of ``profile_phases.py`` (which is
bs_kernel-only). All four methods now forward ``return_phase_timings``
through their wrappers — patched in this same change for paged + mlca,
already there for fasttree + bs_kernel.

Usage:
    uv run python benchmarks/bs_kernel/profile_phases_all.py \\
        --K 64 --L_p 30000 --B 32 --max_new 256
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import dbs, fasttree, mlca, paged
from beam_engine.methods import bs_kernel
from beam_engine.models.modeling_llama import LlamaForCausalLM


import os as _os
MODEL_NAME = _os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")
DEVICE = "cuda"
DTYPE = torch.float16

METHODS = {
    "paged":    paged.beam_search,
    "mlca":     mlca.beam_search,
    "fasttree": fasttree.beam_search,
    "bs_kernel": bs_kernel.beam_search,
    # Diverse Beam Search variants (same kernels, diversity-penalised top-K)
    "dbs_paged":    dbs.dbs_paged,
    "dbs_mlca":     dbs.dbs_mlca,
    "dbs_fasttree": dbs.dbs_fasttree,
    "dbs_bs_kernel": dbs.dbs_bs_kernel,
}


def _make_prompt(tok, target_len: int) -> list[int]:
    seed = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
    )
    ids: list[int] = []
    while len(ids) < target_len:
        ids.extend(tok.encode(seed, add_special_tokens=False))
    return ids[:target_len]


def _fmt_phase_rows(steps_ms, timings):
    rows = []
    total_decode = sum(steps_ms)
    for phase in ("cow", "plan", "forward", "topk", "fork"):
        xs = timings.get(f"{phase}_ms", [])
        if not xs:
            continue
        mean = sum(xs) / len(xs)
        median = statistics.median(xs)
        total = sum(xs)
        pct = 100.0 * total / total_decode if total_decode > 0 else 0.0
        rows.append((phase, mean, median, total, pct))
    return rows


def _print_phase_table(method, prefill_ms, steps_ms, timings):
    n = len(steps_ms)
    total = sum(steps_ms)
    print(f"=== {method} ===")
    print(
        f"  prefill   = {prefill_ms:.1f} ms"
    )
    print(
        f"  decode    = {total:.1f} ms over {n} steps "
        f"({total/max(1,n):.3f} ms/step)"
    )
    rows = _fmt_phase_rows(steps_ms, timings)
    widths = (10, 12, 12, 14, 8)
    header = ("phase", "mean(ms)", "median(ms)", "total(ms)", "%")
    print("  " + "  ".join(f"{h:<{w}}" for h, w in zip(header, widths)))
    for r in rows:
        print(
            f"  {r[0]:<{widths[0]}}  "
            f"{r[1]:<{widths[1]}.4f}  "
            f"{r[2]:<{widths[2]}.4f}  "
            f"{r[3]:<{widths[3]}.2f}  "
            f"{r[4]:<{widths[4]}.1f}"
        )
    print()


def _print_bs_kernel_picks(picks_per_step):
    """picks_per_step: list[Pick] for batch (same pick across prompts)."""
    if not picks_per_step:
        print("  (no picks recorded)")
        return
    keys = []
    for p in picks_per_step:
        if p.strategy.name == "PER_BEAM":
            keys.append(("PER_BEAM", "-", "-", "-"))
        else:
            keys.append(
                (p.strategy.name, str(p.depth), str(p.pool_count), str(p.t_large))
            )
    counts = Counter(keys)
    total = len(picks_per_step)
    print(f"  bs_kernel picks over {total} decode steps:")
    print(f"    {'strategy':<22}  {'depth':<6}  {'pool':<6}  {'t_large':<8}  {'count':<6}  {'pct':<6}")
    for k, c in counts.most_common():
        strat, depth, pool, t_large = k
        pct = 100.0 * c / total
        print(f"    {strat:<22}  {depth:<6}  {pool:<6}  {t_large:<8}  {c:<6}  {pct:<6.1f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--L_p", type=int, default=30000)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument(
        "--methods", nargs="+",
        default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
    )
    args = ap.parse_args()

    print(
        f"cell: K={args.K} L_p={args.L_p} B={args.B} max_new={args.max_new}"
    )
    print(f"methods: {args.methods}\n")

    print("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    base = _make_prompt(tok, args.L_p)
    prompts = [list(base) for _ in range(args.B)]
    page_size = 16
    # All 4 methods share identical-prompt prefix pages via refcount, so
    # only the *distinct* prompts cost prefill pages. With
    # ``[list(base) for _ in range(B)]`` that's 1 set of L_p pages.
    unique_prompts = {tuple(p) for p in prompts}
    unique_prompt_pages = sum(
        (len(p) + page_size - 1) // page_size for p in unique_prompts
    )
    needed_pages = (
        unique_prompt_pages
        + args.B * args.K * ((args.max_new + page_size) // page_size + 2)
        + 256
    )
    print(f"max_num_pages = {needed_pages}\n")

    for name in args.methods:
        fn = METHODS[name]
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        kwargs = dict(
            max_num_pages=needed_pages,
            return_timings=True,
            return_phase_timings=True,
        )
        picks_per_prompt = None
        if name in ("bs_kernel", "dbs_bs_kernel"):
            kwargs["return_picks"] = True
            beams, timings, picks_per_prompt = fn(
                model, config, prompts, args.max_new, args.K, **kwargs,
            )
        else:
            beams, timings = fn(
                model, config, prompts, args.max_new, args.K, **kwargs,
            )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        print(f"({wall:.1f}s wall)")
        _print_phase_table(
            name, timings["prefill_ms"], timings["decode_step_ms"], timings,
        )
        if name in ("bs_kernel", "dbs_bs_kernel") and picks_per_prompt:
            # all prompts share the same pick each step in bs_kernel
            _print_bs_kernel_picks(picks_per_prompt[0])


if __name__ == "__main__":
    main()

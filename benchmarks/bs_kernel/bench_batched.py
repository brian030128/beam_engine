"""Cross-kernel batched-decode benchmark.

For each method × (K, L_p, B), run beam search on B prompts (use
``--distinct`` for pairwise-distinct prompts; default repeats one prompt
B times) and report total decode wall time per (prompt, token). All
methods (paged, tree, fasttree, mlca, adaptive_pool, bs_kernel) are
batched: one decode-loop iteration covers all B prompts, with a single
attention kernel launch per layer covering B·K query rows. The
per-prompt Python loop only builds the layout arrays before they're
concatenated into the batched dispatch.

Output CSV columns:
    method, K, L_p, B, max_new, prefill_ms, decode_total_ms,
    decode_per_token_ms, decode_per_prompt_per_token_ms

Usage:
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 16 --L_p 8192 --B 1 2 4 8 --max_new 16 \
        --methods paged tree fasttree mlca adaptive_pool bs_kernel \
        --out benchmarks/bs_kernel/results/bench_batched.csv
"""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import dbs, deft, fasttree, mlca, paged, tree
from beam_engine.methods import adaptive_pool, bs_kernel
from beam_engine.methods.bs_kernel.cost_model import Strategy
from beam_engine.models import load_model_for_causal_lm
from beam_engine.models.modeling_llama import LlamaForCausalLM


import os as _os
MODEL_NAME = _os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")
DEVICE = "cuda"
DTYPE = torch.float16


def _bs_kernel_force_2l1p(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
):
    """bs_kernel pinned to SHARED_2L_1POOL — adaptive_pool's strategy
    dispatched through bs_kernel's driver path."""
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        available_strategies={Strategy.SHARED_2L_1POOL},
    )


def _bs_kernel_force_2l_dec_tail(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
):
    """bs_kernel pinned to SHARED_2L_DEC_TAIL — shared prefix prefill +
    CTA_Q=1 decode kernel for per-beam tail + merge_state_in_place.
    Baseline depth for the DECTAIL family in dispatch-space ablations."""
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        available_strategies={Strategy.SHARED_2L_DEC_TAIL},
    )


def _bs_kernel_force_3l1p(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
):
    """bs_kernel pinned to SHARED_3L_1POOL — fused 3-level cascade prefill
    (prefix + intermediate + per-beam tail) on the prefill kernel path
    (CTA_Q=K). When a step's workload has no intermediate-level structure,
    the picker's fallback path collapses to SHARED_2L_1POOL."""
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        available_strategies={Strategy.SHARED_3L_1POOL},
    )


def _bs_kernel_force_3l_dec_tail(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
):
    """bs_kernel pinned to SHARED_3L_DEC_TAIL. When a step's workload
    doesn't support depth=3 (no intermediate-level structure), the
    picker's fallback path collapses to the deepest supported family
    (SHARED_2L_DEC_TAIL)."""
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        available_strategies={Strategy.SHARED_3L_DEC_TAIL},
    )


def _bs_kernel_force_4l1p(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
):
    """bs_kernel pinned to SHARED_4L_1POOL. Requires
    ``max_cascade_levels=4`` (wrapper allocation) and
    ``max_dispatch_depth=4`` in the coefficients (picker enumeration).
    Workloads without a depth-4 intermediate structure fall back to
    SHARED_3L_1POOL / SHARED_2L_1POOL via the picker's auto-fallback."""
    from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
    from dataclasses import replace
    coeffs = replace(load_or_defaults("cuda"), max_dispatch_depth=4)
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        max_cascade_levels=4,
        coefficients=coeffs,
        available_strategies={
            Strategy.SHARED_4L_1POOL,
            Strategy.SHARED_3L_1POOL,
            Strategy.SHARED_2L_1POOL,
        },
    )


def _bs_kernel_force_4l_dec_tail(
    model, config, prompts, max_new_tokens, beam_width,
    *, return_timings: bool = False, max_num_pages: int = 2048,
    return_phase_timings: bool = False,
):
    """bs_kernel pinned to SHARED_4L_DEC_TAIL with shallower fallbacks."""
    from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
    from dataclasses import replace
    coeffs = replace(load_or_defaults("cuda"), max_dispatch_depth=4)
    return bs_kernel.beam_search(
        model, config, prompts, max_new_tokens, beam_width,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        max_num_pages=max_num_pages,
        max_cascade_levels=4,
        coefficients=coeffs,
        available_strategies={
            Strategy.SHARED_4L_DEC_TAIL,
            Strategy.SHARED_3L_DEC_TAIL,
            Strategy.SHARED_2L_DEC_TAIL,
        },
    )


METHODS: dict[str, Callable] = {
    "paged":            paged.beam_search,
    "tree":             tree.beam_search,
    "fasttree":         fasttree.beam_search,
    "deft":             deft.beam_search,
    "mlca":             mlca.beam_search,
    "adaptive_pool":    adaptive_pool.beam_search,
    "bs_kernel":        bs_kernel.beam_search,
    "bs_kernel_2l1p":   _bs_kernel_force_2l1p,
    "bs_kernel_2ldt":   _bs_kernel_force_2l_dec_tail,
    "bs_kernel_3l1p":   _bs_kernel_force_3l1p,
    "bs_kernel_3ldt":   _bs_kernel_force_3l_dec_tail,
    "bs_kernel_4l1p":   _bs_kernel_force_4l1p,
    "bs_kernel_4ldt":   _bs_kernel_force_4l_dec_tail,
    # DBS variants — same kernel, diversity-penalized top-K
    # (num_groups=4, λ=0.5 default).
    "dbs_paged":         dbs.dbs_paged,
    "dbs_tree":          dbs.dbs_tree,
    "dbs_fasttree":      dbs.dbs_fasttree,
    "dbs_mlca":          dbs.dbs_mlca,
    "dbs_adaptive_pool": dbs.dbs_adaptive_pool,
    "dbs_bs_kernel":     dbs.dbs_bs_kernel,
}


@dataclass
class Row:
    method: str
    K: int
    L_p: int
    B: int
    max_new: int
    prefill_ms: float
    decode_total_ms: float
    decode_per_token_ms: float
    decode_per_prompt_per_token_ms: float
    # Per-phase decode-step breakdowns (from return_phase_timings=True).
    plan_mean_ms: float = 0.0
    plan_total_ms: float = 0.0
    forward_mean_ms: float = 0.0
    forward_total_ms: float = 0.0
    cow_mean_ms: float = 0.0
    cow_total_ms: float = 0.0
    topk_mean_ms: float = 0.0
    topk_total_ms: float = 0.0
    fork_mean_ms: float = 0.0
    fork_total_ms: float = 0.0


def _phase_mean_total(xs):
    if not xs:
        return 0.0, 0.0
    total = sum(xs)
    return total / len(xs), total


def _make_prompt(tokenizer, target_len: int) -> list[int]:
    snippet = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
        "She traveled across mountains and rivers, recording her findings "
        "in a leather-bound journal that grew thicker with every passing "
        "season. "
    )
    repeats = max(8, target_len // 50)
    while True:
        ids = tokenizer.encode(snippet * repeats, add_special_tokens=False)
        if len(ids) >= target_len:
            break
        repeats *= 2
    return ids[:target_len]


# Distinct seed openings — keep B prompts pairwise prefix-disjoint so
# paged-style methods cannot share prefix pages across prompts.
_DISTINCT_SEEDS = [
    "Once upon a time, in a kingdom far away, there lived a curious scholar who studied the stars. ",
    "The bustling marketplace of Constantinople was filled with merchants from every corner of the known world. ",
    "Deep in the Amazon rainforest, biologists discovered a new species of luminescent frog. ",
    "On the dusty plains of the American Midwest, a young farmer dreamed of becoming a railroad engineer. ",
    "In Edo-period Japan, a master swordsmith forged blades that were said to sing when drawn. ",
    "The Antarctic research station crackled with radio static as the long polar night began. ",
    "Aboard the steamship bound for Liverpool, the elderly diplomat reread the encrypted message. ",
    "High in the Andes, a llama herder noticed strange patterns carved into the volcanic rock. ",
    "The detective lit her pipe and stared at the rain streaking down the office window. ",
    "Long before the first cities rose along the Tigris, hunter-gatherers followed seasonal herds. ",
    "When the great library of Alexandria still stood, scholars argued about the shape of the heavens. ",
    "The Viking longship cut through the cold North Sea waters under a sky heavy with gulls. ",
    "In a sleepy New England fishing village, the lighthouse keeper recorded the day's tide. ",
    "Beneath the towering canopy of redwoods, a paleobotanist sifted through compressed peat. ",
    "The neon-lit streets of Shibuya pulsed with commuters hurrying through the spring drizzle. ",
    "Aboard the International Space Station, the flight engineer floated past the Cupola windows. ",
    "In the dim lamplight of the medieval scriptorium, the monk dipped his quill once more. ",
    "The desert caravan paused at the oasis as the sun began its descent toward the dunes. ",
    "On a quiet research vessel in the Pacific, the marine biologist tagged her hundredth manta ray. ",
    "Through the crowded bazaar of old Damascus, the spice merchant called out his prices. ",
    "The astronaut adjusted her helmet visor and stepped onto the regolith of the lunar far side. ",
    "Inside the cathedral, the organist practiced a fugue while the masons repaired the buttress. ",
    "The Mongolian steppes stretched endlessly under a sky so wide it felt like another ocean. ",
    "Far below the surface of Europa, autonomous probes mapped the hydrothermal vents. ",
    "The Parisian café hummed with conversation about the latest exhibition at the Salon. ",
    "On a rocky coast of Cornwall, the lighthouse keeper recounted tales of shipwrecks past. ",
    "The Silk Road merchant unloaded his bolts of silk at the gates of Samarkand. ",
    "In the Brazilian favela, a young girl practiced her violin on the rooftop each evening. ",
    "The cartographer unrolled his maps and pointed to a coastline no European had yet seen. ",
    "Deep in the Carpathian mountains, the wolf packs moved silently through the winter snow. ",
    "Beneath the Antarctic ice shelf, the autonomous submarine recorded a never-before-heard call. ",
    "The royal astronomer of the Mughal court adjusted his sextant and watched Jupiter rise. ",
]


def _make_distinct_prompts(tokenizer, target_len: int, B: int) -> list[list[int]]:
    """Build B prompts that are pairwise prefix-disjoint.

    Each prompt starts with a unique opening sentence + an index tag, then
    grows to ``target_len`` tokens by repeating its own seed. Different
    prompts therefore diverge from token 0.
    """
    prompts: list[list[int]] = []
    for i in range(B):
        seed = f"Document {i:03d}. " + _DISTINCT_SEEDS[i % len(_DISTINCT_SEEDS)]
        repeats = max(8, target_len // 50)
        while True:
            ids = tokenizer.encode(seed * repeats, add_special_tokens=False)
            if len(ids) >= target_len:
                break
            repeats *= 2
        prompts.append(ids[:target_len])
    return prompts


def run_one(
    method_name: str,
    method_fn: Callable,
    model,
    config,
    prompts: list[list[int]],
    K: int,
    max_new: int,
    max_pages_override: int | None = None,
) -> Row | None:
    B = len(prompts)
    L_p = len(prompts[0])
    page_size = 16
    needed_pages = (
        sum((len(p) + page_size - 1) // page_size for p in prompts)
        + B * K * ((max_new + page_size) // page_size + 2)
        + 256
    )
    if max_pages_override is not None:
        needed_pages = max_pages_override
    extra_kwargs: dict = {}
    sig = inspect.signature(method_fn)
    if "max_num_pages" in sig.parameters:
        extra_kwargs["max_num_pages"] = needed_pages

    if "return_phase_timings" in sig.parameters:
        extra_kwargs["return_phase_timings"] = True

    try:
        beams, timings = method_fn(
            model, config, prompts, max_new, K,
            return_timings=True, **extra_kwargs,
        )
    except Exception as e:
        # OOM ends up here too; aggressive empty_cache to release any partial
        # buffers the failed driver allocated before the exception bubbled up.
        print(f"  [{method_name}] FAILED: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()
        return None

    decode_steps = timings["decode_step_ms"]
    decode_total = sum(decode_steps)
    n_steps = len(decode_steps)
    # All methods are batched now: n_steps == max_new - 1 regardless of B.
    # Per-(prompt,token) latency normalizes by total tokens decoded
    # (B * (max_new - 1)).
    total_tokens = B * (max_new - 1)
    per_token = decode_total / max(1, total_tokens)
    plan_mean, plan_total = _phase_mean_total(timings.get("plan_ms", []))
    fwd_mean,  fwd_total  = _phase_mean_total(timings.get("forward_ms", []))
    cow_mean,  cow_total  = _phase_mean_total(timings.get("cow_ms", []))
    topk_mean, topk_total = _phase_mean_total(timings.get("topk_ms", []))
    fork_mean, fork_total = _phase_mean_total(timings.get("fork_ms", []))
    return Row(
        method=method_name,
        K=K, L_p=L_p, B=B, max_new=max_new,
        prefill_ms=timings["prefill_ms"],
        decode_total_ms=decode_total,
        decode_per_token_ms=decode_total / max(1, n_steps),
        decode_per_prompt_per_token_ms=per_token,
        plan_mean_ms=plan_mean, plan_total_ms=plan_total,
        forward_mean_ms=fwd_mean, forward_total_ms=fwd_total,
        cow_mean_ms=cow_mean, cow_total_ms=cow_total,
        topk_mean_ms=topk_mean, topk_total_ms=topk_total,
        fork_mean_ms=fork_mean, fork_total_ms=fork_total,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--L_p", nargs="+", type=int, default=[8192])
    ap.add_argument("--B", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    ap.add_argument("--out", default=None)
    ap.add_argument("--distinct", action="store_true",
                    help="use B pairwise-distinct prompts instead of B copies "
                         "of one prompt (disables cross-prompt page sharing)")
    ap.add_argument("--max_pages", type=int, default=None,
                    help="override the computed max_num_pages cap (slab "
                         "allocation size). Use for shared-prompt cells where "
                         "the worst-case bound overallocates.")
    ap.add_argument("--warmup", action="store_true",
                    help="run each (method, cell) twice and discard the "
                         "first iter (absorbs Triton-JIT autotune + first-"
                         "call CUDA-graph capture).")
    ap.add_argument("--prompts-file", default=None,
                    help="JSONL file (one record per line, fields "
                         "{token_len, prompt}) supplying real-world prompts. "
                         "First B records are taken (skipping any with "
                         "token_len < L_p) and truncated to L_p tokens. "
                         "Overrides --distinct.")
    args = ap.parse_args()

    methods = {k: METHODS[k] for k in args.methods if k in METHODS}
    if not methods:
        print(f"no valid methods in {args.methods}; valid: {list(METHODS.keys())}", file=sys.stderr)
        sys.exit(2)

    grid = [(K, L_p, B) for K in args.K for L_p in args.L_p for B in args.B]
    print(f"grid: {len(grid)} (K,L_p,B) cells × {len(methods)} methods = {len(grid) * len(methods)} runs")
    print(f"methods: {list(methods.keys())}")
    print(f"max_new: {args.max_new}")

    print("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model_for_causal_lm(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    # Pre-load real-world prompts once if --prompts-file is set; we
    # re-truncate per (K, L_p, B) cell below.
    file_prompts_raw: list[list[int]] | None = None
    if args.prompts_file:
        import json as _json
        file_prompts_raw = []
        with open(args.prompts_file) as _pf:
            for line in _pf:
                rec = _json.loads(line)
                # Tokenize once; we'll truncate per cell.
                file_prompts_raw.append(tok.encode(rec["prompt"]))
        print(f"loaded {len(file_prompts_raw)} prompts from {args.prompts_file}")

    rows: list[Row] = []
    for (K, L_p, B) in grid:
        if file_prompts_raw is not None:
            usable = [ids for ids in file_prompts_raw if len(ids) >= L_p]
            if len(usable) < B:
                print(f"  WARNING: only {len(usable)} prompts ≥ {L_p} tokens; "
                      f"need {B}. Skipping (K={K}, L_p={L_p}, B={B}).")
                continue
            prompts = [usable[i][:L_p] for i in range(B)]
        elif args.distinct:
            prompts = _make_distinct_prompts(tok, L_p, B)
        else:
            base = _make_prompt(tok, L_p)
            prompts = [list(base) for _ in range(B)]
        for name, fn in methods.items():
            iters = 2 if args.warmup else 1
            for it in range(iters):
                tag = "warmup " if (args.warmup and it == 0) else "       "
                print(f"  {name:<14} {tag}K={K:<3d} L_p={L_p:<6d} B={B:<2d}", end=" ", flush=True)
                t0 = time.perf_counter()
                row = run_one(name, fn, model, config, prompts, K, args.max_new,
                              max_pages_override=args.max_pages)
                t1 = time.perf_counter()
                if row is None:
                    torch.cuda.empty_cache()
                    break  # OOM / failure — don't bother retrying
                # Discard warmup iteration's row.
                if args.warmup and it == 0:
                    print(
                        f"discard          decode_total={row.decode_total_ms:7.1f}ms  "
                        f"({t1 - t0:5.1f}s wall)"
                    )
                else:
                    rows.append(row)
                    print(
                        f"prefill={row.prefill_ms:7.1f}ms  "
                        f"decode_total={row.decode_total_ms:7.1f}ms  "
                        f"per_pt_token={row.decode_per_prompt_per_token_ms:6.2f}ms  "
                        f"({t1 - t0:5.1f}s wall)"
                    )
                torch.cuda.empty_cache()

    out = args.out
    if out is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        gpu = torch.cuda.get_device_name().replace(" ", "_")
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = results_dir / f"bench_batched-{gpu}-{ts}.csv"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "K", "L_p", "B", "max_new",
            "prefill_ms", "decode_total_ms",
            "decode_per_token_ms", "decode_per_prompt_per_token_ms",
            "plan_mean_ms", "plan_total_ms",
            "forward_mean_ms", "forward_total_ms",
            "cow_mean_ms", "cow_total_ms",
            "topk_mean_ms", "topk_total_ms",
            "fork_mean_ms", "fork_total_ms",
        ])
        for r in rows:
            w.writerow([
                r.method, r.K, r.L_p, r.B, r.max_new,
                f"{r.prefill_ms:.4f}",
                f"{r.decode_total_ms:.4f}",
                f"{r.decode_per_token_ms:.4f}",
                f"{r.decode_per_prompt_per_token_ms:.4f}",
                f"{r.plan_mean_ms:.4f}",    f"{r.plan_total_ms:.4f}",
                f"{r.forward_mean_ms:.4f}", f"{r.forward_total_ms:.4f}",
                f"{r.cow_mean_ms:.4f}",     f"{r.cow_total_ms:.4f}",
                f"{r.topk_mean_ms:.4f}",    f"{r.topk_total_ms:.4f}",
                f"{r.fork_mean_ms:.4f}",    f"{r.fork_total_ms:.4f}",
            ])
    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

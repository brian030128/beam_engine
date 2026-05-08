"""Phase-timing comparison: fasttree vs bs_kernel at the canonical cell.

Reports per-step mean/median/total for each phase (cow, plan, forward,
topk, fork). "forward" is the model.forward call (q/k/v projection +
attention + FFN + lm_head). The attention kernel itself is the
dominant term within "forward" for long-context decode, so a smaller
"forward" total = faster attend kernel + write path.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from transformers import AutoTokenizer

from beam_engine.baselines.fasttree import beam_search as fasttree_search
from beam_engine.methods.bs_kernel import beam_search as bs_kernel_search
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda"
DTYPE = torch.float16

METHODS = {
    "fasttree": fasttree_search,
    "bs_kernel": bs_kernel_search,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--L_p", type=int, default=8192)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--methods", nargs="+", default=["fasttree", "bs_kernel"])
    args = ap.parse_args()

    print(f"K={args.K} L_p={args.L_p} B={args.B} max_new={args.max_new}")
    print(f"methods: {args.methods}\n")

    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device=DEVICE,
    )
    config = model.config
    base = _make_prompt(tok, args.L_p)
    prompts = [list(base) for _ in range(args.B)]
    page_size = 16
    needed_pages = (
        sum((len(p) + page_size - 1) // page_size for p in prompts)
        + args.B * args.K * ((args.max_new + page_size) // page_size + 2)
        + 256
    )
    print("Model loaded.\n")

    for name in args.methods:
        fn = METHODS[name]
        t0 = time.perf_counter()
        beams, timings = fn(
            model, config, prompts, args.max_new, args.K,
            max_num_pages=needed_pages,
            return_timings=True,
            return_phase_timings=True,
        )
        wall = time.perf_counter() - t0
        steps = timings["decode_step_ms"]
        n = len(steps)
        per_pt = sum(steps) / (args.B * (args.max_new - 1))
        print(
            f"=== method={name}  ({n} decode steps, {wall:.1f}s wall, "
            f"per_pt_token={per_pt:.3f} ms) ==="
        )
        print(
            f"  total_decode = {sum(steps):.1f} ms  "
            f"({sum(steps)/n:.2f} ms/step)"
        )
        rows = []
        for phase in ("cow", "plan", "forward", "topk", "fork"):
            xs = timings.get(f"{phase}_ms", [])
            if not xs:
                continue
            mean = sum(xs) / len(xs)
            median = statistics.median(xs)
            total = sum(xs)
            pct = 100.0 * total / sum(steps) if sum(steps) > 0 else 0.0
            rows.append((phase, mean, median, total, pct))

        col_widths = (10, 10, 10, 12, 8)
        header = ("phase", "mean(ms)", "median(ms)", "total(ms)", "%")
        print("  " + "  ".join(f"{h:<{w}}" for h, w in zip(header, col_widths)))
        for r in rows:
            print(
                f"  {r[0]:<{col_widths[0]}}  "
                f"{r[1]:<{col_widths[1]}.3f}  "
                f"{r[2]:<{col_widths[2]}.3f}  "
                f"{r[3]:<{col_widths[3]}.1f}  "
                f"{r[4]:<{col_widths[4]}.1f}"
            )
        print()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

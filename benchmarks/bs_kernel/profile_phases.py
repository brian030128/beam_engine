"""Per-phase timing breakdown for one bs_kernel decode at K=64/L_p=8192/B=32.

Phases:
  cow      page-table copy-on-write (host bookkeeping + GPU memcpy when forks)
  plan     backend.plan_decode_step (build qo/kv/lpl tensors, wrapper.plan)
  forward  model.forward (q/k/v projections + attention + FFN + lm_head + log_softmax)
  topk     standard top-K selection
  fork     refcount surgery + new beam list construction

Reports mean / median / sum across all decode steps for the picker, and
optionally for forced modes via --modes.

Each phase uses torch.cuda.synchronize() barriers so reported per-phase
times include host-visible kernel completion time. Total step time
shown alongside, which sums to slightly more than the un-instrumented
baseline due to the sync barriers themselves (~5 µs each × 4 = ~20 µs/step).
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from transformers import AutoTokenizer

from beam_engine.methods.bs_kernel import beam_search as bs_kernel_search
from beam_engine.methods.bs_kernel.cost_model import Strategy
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda"
DTYPE = torch.float16


MODES = {
    "picker": None,
    "SHARED_2L_1POOL": {Strategy.SHARED_2L_1POOL},
    "SHARED_3L_1POOL": {Strategy.SHARED_3L_1POOL},
    "SHARED_2L_DEC_TAIL": {Strategy.SHARED_2L_DEC_TAIL},
    "PER_BEAM": {Strategy.PER_BEAM},
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
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--modes", nargs="+", default=["picker", "SHARED_2L_1POOL"])
    args = ap.parse_args()

    print(f"K={args.K} L_p={args.L_p} B={args.B} max_new={args.max_new}")
    print(f"modes: {args.modes}\n")

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

    for mode in args.modes:
        available = MODES.get(mode)
        kwargs = {
            "max_num_pages": needed_pages,
            "return_timings": True,
            "return_phase_timings": True,
        }
        if available is not None:
            kwargs["available_strategies"] = available
        t0 = time.perf_counter()
        beams, timings = bs_kernel_search(
            model, config, prompts, args.max_new, args.K, **kwargs,
        )
        wall = time.perf_counter() - t0
        steps = timings["decode_step_ms"]
        n = len(steps)
        print(f"=== mode={mode}  ({n} decode steps, {wall:.1f}s wall) ===")
        print(
            f"  total_decode = {sum(steps):.1f} ms  "
            f"({sum(steps)/n:.2f} ms/step  "
            f"per_pt_token={sum(steps)/(args.B * (args.max_new-1)):.3f} ms)"
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

"""Side-by-side plan-phase breakdown for paged vs bs_kernel.

Runs both methods at two representative cells (a PER_BEAM cell where
paged wins, and a DEC_TAIL cell where bs_kernel wins) with per-phase
trace enabled. Each method prints its own breakdown.

Usage:
    PAGED_TRACE_PLAN=1 BS_KERNEL_TRACE_PLAN=1 \\
        uv run python benchmarks/bs_kernel/profile_plan_breakdown.py
"""

from __future__ import annotations

import os
import time

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import paged
from beam_engine.methods import bs_kernel
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")
DEVICE = "cuda"
DTYPE = torch.float16


CELLS = [
    # (K, L_p, B, max_new, label) — picked to exercise both bs_kernel paths
    (4,  8192,  1,  256, "PER_BEAM regime (paged wins)"),
    (64, 30000, 16, 256, "DEC_TAIL regime (bs_kernel wins)"),
]


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
    print("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    for K, L_p, B, max_new, label in CELLS:
        print(f"\n{'='*70}\nCell: K={K} L_p={L_p} B={B} max_new={max_new} — {label}\n{'='*70}")
        base = _make_prompt(tok, L_p)
        prompts = [list(base) for _ in range(B)]
        page_size = 16
        unique_prompts = {tuple(p) for p in prompts}
        unique_prompt_pages = sum(
            (len(p) + page_size - 1) // page_size for p in unique_prompts
        )
        needed_pages = (
            unique_prompt_pages
            + B * K * ((max_new + page_size) // page_size + 2)
            + 256
        )

        # ---- paged ----
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, timings = paged.beam_search(
            model, config, prompts, max_new, K,
            max_num_pages=needed_pages,
            return_timings=True, return_phase_timings=True,
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        steps = timings["decode_step_ms"]
        n = len(steps)
        plan_mean = sum(timings["plan_ms"]) / max(1, len(timings["plan_ms"]))
        fwd_mean = sum(timings["forward_ms"]) / max(1, len(timings["forward_ms"]))
        ms_step = sum(steps) / max(1, n)
        print(
            f"\n  paged: ms/step={ms_step:.2f}  plan={plan_mean:.2f}  "
            f"fwd={fwd_mean:.2f}  wall={wall:.1f}s"
        )

        # ---- bs_kernel ----
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, timings = bs_kernel.beam_search(
            model, config, prompts, max_new, K,
            max_num_pages=needed_pages,
            return_timings=True, return_phase_timings=True,
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        steps = timings["decode_step_ms"]
        n = len(steps)
        plan_mean = sum(timings["plan_ms"]) / max(1, len(timings["plan_ms"]))
        fwd_mean = sum(timings["forward_ms"]) / max(1, len(timings["forward_ms"]))
        ms_step = sum(steps) / max(1, n)
        print(
            f"\n  bs_kernel: ms/step={ms_step:.2f}  plan={plan_mean:.2f}  "
            f"fwd={fwd_mean:.2f}  wall={wall:.1f}s"
        )


if __name__ == "__main__":
    main()

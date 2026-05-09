"""Decompose fasttree's plan phase into sub-steps.

Splits ``plan_decode_step`` into:
  - gather_paths         : our integration shim (read beam.pages_*)
  - build_radix_tree     : page-level radix construction (our shim)
  - expand_to_slots      : page → slot expansion (our shim)
  - build_metadata       : FastTree's algorithm (_tree_heuristic +
                           _compute_parallelism + vnode tensor build)
  - write_slots          : our shim (Python loop + torch.tensor)
  - out_buf              : output buffer alloc/reuse

Tells us how much of plan time is our integration overhead vs how much is
intrinsic to FastTree's per-step routing decision.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoTokenizer

from beam_engine.baselines.fasttree import (
    beam_search as fasttree_search,
    get_plan_subtimes,
    reset_plan_subtimes,
)
from beam_engine.models.modeling_llama import LlamaForCausalLM


import os as _os
MODEL_NAME = _os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")


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
    K, L_p, B, max_new = 64, 8192, 32, 256
    print(f"K={K} L_p={L_p} B={B} max_new={max_new}")

    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device="cuda",
    )
    config = model.config
    base = _make_prompt(tok, L_p)
    prompts = [list(base) for _ in range(B)]
    page_size = 16
    needed_pages = (
        sum((len(p) + page_size - 1) // page_size for p in prompts)
        + B * K * ((max_new + page_size) // page_size + 2)
        + 256
    )
    print("Model loaded.\n")

    reset_plan_subtimes()
    t0 = time.perf_counter()
    beams, timings = fasttree_search(
        model, config, prompts, max_new, K,
        max_num_pages=needed_pages,
        return_timings=True,
        return_phase_timings=True,
    )
    wall = time.perf_counter() - t0

    steps = timings["decode_step_ms"]
    n = len(steps)
    plan_total = sum(timings["plan_ms"])
    forward_total = sum(timings["forward_ms"])

    print(f"=== fasttree ({n} steps, {wall:.1f}s wall) ===")
    print(f"  total_decode = {sum(steps):.1f} ms")
    print(f"  plan_total   = {plan_total:.1f} ms  ({plan_total/n:.2f} ms/step)")
    print(f"  forward      = {forward_total:.1f} ms  "
          f"({forward_total/n:.2f} ms/step)")
    sub = get_plan_subtimes()
    sub_sum = sum(sub.values())
    print(f"\n  --- plan sub-steps (sum={sub_sum:.1f} ms, "
          f"{100*sub_sum/plan_total:.1f}% of plan) ---")
    rows = sorted(sub.items(), key=lambda kv: -kv[1])
    for k, v in rows:
        print(f"  {k:<22s} {v:>10.1f} ms  "
              f"({v/n:>6.3f} ms/step, {100*v/plan_total:>5.1f}% of plan)")


if __name__ == "__main__":
    main()

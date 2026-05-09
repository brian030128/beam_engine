"""Time bs_kernel at K=64/L_p=8K/B=32, splitting forward into:
  attn  — total time inside ctx.attend across all 16 layers per step
  rest  — model.forward minus attn (proj + FFN + lm_head + log_softmax)

Run with ``BE_PROFILE_ATTN=1`` to enable the per-attend sync timer.
"""
from __future__ import annotations

import os
import time

import torch
from transformers import AutoTokenizer

from beam_engine.methods.bs_kernel import beam_search as bs_kernel_search
from beam_engine.models.attention import (
    get_attn_subtimes,
    reset_attn_subtimes,
)
from beam_engine.models.modeling_llama import LlamaForCausalLM


def _make_prompt(tok, target_len):
    seed = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
    )
    ids = []
    while len(ids) < target_len:
        ids.extend(tok.encode(seed, add_special_tokens=False))
    return ids[:target_len]


def main():
    K = int(os.environ.get("BE_K", 64))
    L_p = int(os.environ.get("BE_L_P", 8192))
    B = int(os.environ.get("BE_B", 32))
    max_new = int(os.environ.get("BE_MAX_NEW", 256))
    print(f"K={K} L_p={L_p} B={B} max_new={max_new}")
    print(f"BE_PROFILE_ATTN={os.environ.get('BE_PROFILE_ATTN')}")

    model_name = os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")
    print(f"model={model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device="cuda",
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

    reset_attn_subtimes()
    t0 = time.perf_counter()
    beams, timings = bs_kernel_search(
        model, config, prompts, max_new, K,
        max_num_pages=needed_pages,
        return_timings=True,
        return_phase_timings=True,
    )
    wall = time.perf_counter() - t0

    steps = timings["decode_step_ms"]
    n = len(steps)
    forward_total = sum(timings["forward_ms"])
    plan_total = sum(timings["plan_ms"])
    cow_total = sum(timings["cow_ms"])
    topk_total = sum(timings["topk_ms"])
    fork_total = sum(timings["fork_ms"])
    step_total = sum(steps)
    attn_total = get_attn_subtimes()["attn_ms"]
    rest = forward_total - attn_total

    print(f"\n=== bs_kernel ({n} steps, {wall:.1f}s wall, "
          f"per_pt_token={step_total/(B*(max_new-1)):.3f} ms) ===")
    print(f"  total step    = {step_total:.1f} ms ({step_total/n:.3f} ms/step)")
    print()
    print(f"{'phase':<12} {'ms/step':>10} {'%':>6}")
    rows = [
        ("plan",     plan_total/n,     100*plan_total/step_total),
        ("forward",  forward_total/n,  100*forward_total/step_total),
        ("  attn",   attn_total/n,     100*attn_total/step_total),
        ("  rest",   rest/n,           100*rest/step_total),
        ("topk",     topk_total/n,     100*topk_total/step_total),
        ("cow",      cow_total/n,      100*cow_total/step_total),
        ("fork",     fork_total/n,     100*fork_total/step_total),
    ]
    for name, ms, pct in rows:
        print(f"{name:<12} {ms:>10.3f} {pct:>5.1f}%")


if __name__ == "__main__":
    main()

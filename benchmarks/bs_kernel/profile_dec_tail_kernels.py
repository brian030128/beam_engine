"""Profile DEC_TAIL sub-kernels vs SHARED_2L_1POOL cascade kernel.

Forces each strategy at the K=16/B=16/L_p=8192 cell where DEC_TAIL
empirically loses to 1POOL by ~5 ms/step. With per-sub-kernel CUDA
event timing enabled (BS_KERNEL_TRACE_DEC_TAIL_KERNELS=1 and
BS_KERNEL_TRACE_CASCADE_KERNELS=1), reports:

  * DEC_TAIL: append_ms + tail_ms + prefix_ms + merge_ms per layer
  * 1POOL:    append_ms + cascade_ms per layer

This tells us whether the gap is:
  (a) in prefix_ms (SM underutilization / kernel-fit at packed_qo=64
      against a very long shared KV), or
  (b) in merge_ms (epilogue scatter), or
  (c) launch-overhead between the multiple sub-kernels.

Usage:
    uv run python benchmarks/bs_kernel/profile_dec_tail_kernels.py
"""

from __future__ import annotations

import os
import statistics
import time

import torch
from transformers import AutoTokenizer

# Set trace env vars BEFORE importing bs_kernel so any module-level
# reads are correct.
os.environ.setdefault("BS_KERNEL_TRACE_DEC_TAIL_KERNELS", "1")
os.environ.setdefault("BS_KERNEL_TRACE_CASCADE_KERNELS", "1")

from beam_engine.methods import bs_kernel
from beam_engine.methods.bs_kernel.cost_model import Strategy
from beam_engine.methods.bs_kernel.decode_tail_context import (
    DEC_TAIL_KERNEL_TRACE,
)
from beam_engine.methods.adaptive_pool import CASCADE_KERNEL_TRACE
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")
DEVICE = "cuda"
DTYPE = torch.float16
PAGE_SIZE = 16


def _make_prompt(tok, target_len: int) -> list[int]:
    seed = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
    )
    ids: list[int] = []
    while len(ids) < target_len:
        ids.extend(tok.encode(seed, add_special_tokens=False))
    return ids[:target_len]


def _summary(trace: list[dict], drop_first_step_layers: int = 16) -> dict:
    """Median of each sub-kernel time across captured calls. Skip the
    first ``drop_first_step_layers`` entries (= the very first decode
    step's 16 layer calls) to drop launch-time warmups.
    """
    if not trace:
        return {}
    rows = trace[drop_first_step_layers:] if len(trace) > drop_first_step_layers else trace
    keys = [k for k in rows[0].keys() if k != "layer_idx"]
    out: dict[str, float] = {}
    for k in keys:
        out[k] = statistics.median(r[k] for r in rows)
    out["_n"] = len(rows)
    return out


def main():
    K, L_p, B, max_new = 16, 8192, 16, 64

    print(f"Loading model {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device=DEVICE,
    )
    config = model.config
    print(f"model: hidden_size={config.hidden_size} "
          f"num_qo_heads={config.num_attention_heads} "
          f"num_kv_heads={config.num_key_value_heads} "
          f"num_layers={config.num_hidden_layers}\n")

    base = _make_prompt(tok, L_p)
    prompts = [list(base) for _ in range(B)]
    unique_pages = (L_p + PAGE_SIZE - 1) // PAGE_SIZE
    needed_pages = (
        unique_pages + B * K * ((max_new + PAGE_SIZE) // PAGE_SIZE + 2) + 256
    )
    print(f"=== K={K} L_p={L_p} B={B}  max_new={max_new}  pages={needed_pages} ===\n")

    # --- DEC_TAIL ---
    DEC_TAIL_KERNEL_TRACE.clear()
    print("Running bs_kernel forced to SHARED_2L_DEC_TAIL ...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = bs_kernel.beam_search(
        model, config, prompts, max_new, K,
        max_num_pages=needed_pages,
        return_timings=True,
        return_phase_timings=True,
        available_strategies={Strategy.SHARED_2L_DEC_TAIL},
    )
    torch.cuda.synchronize()
    print(f"  wall: {time.perf_counter()-t0:.1f}s  layer-traces: {len(DEC_TAIL_KERNEL_TRACE)}\n")

    # --- 1POOL ---
    CASCADE_KERNEL_TRACE.clear()
    print("Running bs_kernel forced to SHARED_2L_1POOL ...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = bs_kernel.beam_search(
        model, config, prompts, max_new, K,
        max_num_pages=needed_pages,
        return_timings=True,
        return_phase_timings=True,
        available_strategies={Strategy.SHARED_2L_1POOL},
    )
    torch.cuda.synchronize()
    print(f"  wall: {time.perf_counter()-t0:.1f}s  layer-traces: {len(CASCADE_KERNEL_TRACE)}\n")

    # --- Report ---
    dt = _summary(DEC_TAIL_KERNEL_TRACE)
    cas = _summary(CASCADE_KERNEL_TRACE)

    print("=" * 78)
    print("Per-layer sub-kernel timings (median across captured layer calls)")
    print(f"  K={K} L_p={L_p} B={B}  Q-per-prompt = K·gqa = {K*config.num_attention_heads//config.num_key_value_heads}")
    print("=" * 78)

    print(f"\nDEC_TAIL ({dt.get('_n', 0)} samples)")
    total_dt = (dt.get("append_ms", 0) + dt.get("tail_ms", 0)
                + dt.get("prefix_ms", 0) + dt.get("merge_ms", 0))
    for k in ("append_ms", "tail_ms", "prefix_ms", "merge_ms"):
        v = dt.get(k, 0)
        pct = 100.0 * v / total_dt if total_dt > 0 else 0
        print(f"  {k:>14s} = {v*1000:7.1f} µs  ({pct:4.1f}%)")
    print(f"  {'TOTAL':>14s} = {total_dt*1000:7.1f} µs")

    print(f"\n1POOL ({cas.get('_n', 0)} samples)")
    total_cas = cas.get("append_ms", 0) + cas.get("cascade_ms", 0)
    for k in ("append_ms", "cascade_ms"):
        v = cas.get(k, 0)
        pct = 100.0 * v / total_cas if total_cas > 0 else 0
        print(f"  {k:>14s} = {v*1000:7.1f} µs  ({pct:4.1f}%)")
    print(f"  {'TOTAL':>14s} = {total_cas*1000:7.1f} µs")

    gap = (total_dt - total_cas) * 1000
    print()
    print(f"Per-layer gap (DEC_TAIL − 1POOL): {gap:+.1f} µs")
    print(f"Per-layer gap as fraction of DEC_TAIL: {100.0 * (total_dt - total_cas) / total_dt:+.1f}%")
    print()
    # Attribution: which sub-kernel is the gap?
    cas_attn = cas.get("cascade_ms", 0) * 1000  # us
    dec_attn = (dt.get("tail_ms", 0) + dt.get("prefix_ms", 0) + dt.get("merge_ms", 0)) * 1000
    print("Where the gap lives:")
    print(f"  cascade (1POOL one kernel): {cas_attn:7.1f} µs")
    print(f"  DEC_TAIL three kernels:     {dec_attn:7.1f} µs")
    print(f"    decode tail:              {dt.get('tail_ms', 0)*1000:7.1f} µs")
    print(f"    prefix prefill:           {dt.get('prefix_ms', 0)*1000:7.1f} µs")
    print(f"    merge_state_in_place:     {dt.get('merge_ms', 0)*1000:7.1f} µs")
    print(f"  attention-only gap:         {dec_attn - cas_attn:+7.1f} µs")


if __name__ == "__main__":
    main()

"""Per-layer DEC_TAIL kernel-time breakdown at K=16/B=4/L_p=50K.

Forces DEC_TAIL on the cell that exhibits the +34% picker miss and dumps
mean/median/p99 wall time of each sub-kernel (append_paged_kv_cache,
decode_wrapper.run, prefix_wrapper.run, merge_state_in_place) per layer.

Also runs forced FUSED on the same cell to compare. Net delta in
per-attention-call wall time should explain where the ~400 µs/call gap
between model prediction and measured DEC_TAIL cost actually goes.

Usage:
    BS_KERNEL_TRACE_DEC_TAIL_KERNELS=1 \
    uv run python scripts/paper-exp/probe_dec_tail_layer_breakdown.py
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("BS_KERNEL_TRACE_DEC_TAIL_KERNELS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from beam_engine.methods import bs_kernel
from beam_engine.methods.bs_kernel.cost_model import Strategy
from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
from beam_engine.methods.bs_kernel import decode_tail_context as dtc
from beam_engine.models import load_model_for_causal_lm
from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B"
DEVICE = torch.device("cuda")
DTYPE = torch.float16

K = 16
L_p = 50000
B = 4
MAX_NEW = 256


def synth_prompt(tokenizer, target_len: int) -> list[int]:
    snippet = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
    )
    out = tokenizer.encode(snippet * 1024, add_special_tokens=False)
    while len(out) < target_len:
        out += out
    return out[:target_len]


def run_one(model, config, prompts, strategy, label):
    """Run forced strategy on the cell, return measured timings + trace."""
    print(f"\n=== forcing {label} ({strategy}) on K={K} B={B} L_p={L_p} mn={MAX_NEW} ===")
    coeff = load_or_defaults(DEVICE, MODEL, tp_size=1)
    dtc.DEC_TAIL_KERNEL_TRACE.clear()
    # warmup
    bs_kernel.beam_search(
        model, config, prompts, MAX_NEW, K,
        max_num_pages=20000, available_strategies={strategy},
        coefficients=coeff, dtype=DTYPE, return_timings=False,
    )
    torch.cuda.synchronize()
    dtc.DEC_TAIL_KERNEL_TRACE.clear()
    # measured run
    beams, timings = bs_kernel.beam_search(
        model, config, prompts, MAX_NEW, K,
        max_num_pages=20000, available_strategies={strategy},
        coefficients=coeff, dtype=DTYPE, return_timings=True,
    )
    torch.cuda.synchronize()
    return timings, list(dtc.DEC_TAIL_KERNEL_TRACE)


def summarise_trace(trace: list[dict], label: str):
    if not trace:
        print(f"[{label}] no trace entries (forced strategy must be DEC_TAIL family)")
        return
    keys = ("append_ms", "tail_ms", "prefix_ms", "merge_ms")
    print(f"\n[{label}] per-attention-call sub-kernel times (over {len(trace)} layer×step samples)")
    print(f"  {'kernel':<14}  {'mean (µs)':>10}  {'p50 (µs)':>10}  {'p99 (µs)':>10}  {'total (ms)':>12}")
    for k in keys:
        xs_us = [r[k] * 1000.0 for r in trace]
        xs_sorted = sorted(xs_us)
        mean = sum(xs_us) / len(xs_us)
        p50 = statistics.median(xs_us)
        p99 = xs_sorted[int(0.99 * (len(xs_us) - 1))]
        total_ms = sum(xs_us) / 1000.0
        print(f"  {k:<14}  {mean:>10.2f}  {p50:>10.2f}  {p99:>10.2f}  {total_ms:>12.1f}")
    # Sum of all four per call
    sums_per_call = [sum(r[k] for k in keys) * 1000.0 for r in trace]
    mean_sum = sum(sums_per_call) / len(sums_per_call)
    total_ms = sum(sums_per_call) / 1000.0
    print(f"  {'SUM':<14}  {mean_sum:>10.2f}  {'-':>10}  {'-':>10}  {total_ms:>12.1f}")


def main():
    print(f"loading model {MODEL} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_model_for_causal_lm(MODEL, dtype=DTYPE, device=DEVICE)
    config = model.config
    print(f"model loaded; num_layers={config.num_hidden_layers}, num_kv_heads={config.num_key_value_heads}")

    prompts = [synth_prompt(tok, L_p) for _ in range(B)]
    print(f"built {B} prompts of length ~{len(prompts[0])} tokens")

    # Run forced DEC_TAIL and capture sub-kernel trace
    timings_dt, trace_dt = run_one(model, config, prompts, Strategy.SHARED_2L_DEC_TAIL, "DEC_TAIL")
    summarise_trace(trace_dt, "DEC_TAIL")
    n_layers = config.num_hidden_layers
    n_steps = MAX_NEW - 1
    decode_total_ms = sum(timings_dt["decode_step_ms"])
    per_layer_call_ms = decode_total_ms / max(1, n_steps * n_layers)
    print(
        f"\n[DEC_TAIL] decode_total={decode_total_ms:.1f} ms over "
        f"{n_steps} steps × {n_layers} layers "
        f"= {per_layer_call_ms*1000:.1f} µs/attention-call"
    )

    # Run forced FUSED for comparison (no DEC_TAIL trace populated)
    print("\n--- forced FUSED for comparison ---")
    timings_fu, _ = run_one(model, config, prompts, Strategy.SHARED_2L_1POOL, "FUSED")
    decode_total_fu = sum(timings_fu["decode_step_ms"])
    per_layer_call_fu = decode_total_fu / max(1, n_steps * n_layers)
    print(
        f"\n[FUSED]    decode_total={decode_total_fu:.1f} ms over "
        f"{n_steps} steps × {n_layers} layers "
        f"= {per_layer_call_fu*1000:.1f} µs/attention-call"
    )
    print(
        f"\nΔ per attention call: {(per_layer_call_ms - per_layer_call_fu)*1000:+.1f} µs "
        f"(DEC_TAIL minus FUSED)"
    )


if __name__ == "__main__":
    main()

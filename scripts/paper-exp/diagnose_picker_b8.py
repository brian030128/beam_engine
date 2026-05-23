"""Diagnose the picker's B=8 mispick on 70B-fp8 + fp8 KV.

For each target cell (K, L_p, B):
  - Measure decode_step_ms for every probe mode
  - Compute cost_model's predicted cost for every Strategy with a
    per-term breakdown (bw / compute / merge / decode_launch / decode_bw /
    decode_work). cost_decode_tail and cost_dec_tail printed separately.
  - Print a side-by-side table: model_us | measured_us | ratio
  - Highlight the picker's pick vs the oracle's pick.

Run via torchrun on TP=2 H100 with BE_MODEL/BE_DTYPE/BE_KV_DTYPE set.
"""

from __future__ import annotations

import os
from collections import defaultdict

import torch
from transformers import AutoTokenizer

from beam_engine.distributed import (
    destroy_tp, get_tp_rank, get_tp_world_size, init_tp,
)
from beam_engine.models import load_model_for_causal_lm
from beam_engine.methods.bs_kernel.autotune import (
    _measure_strategy_times, _make_workload, MODE_FILTERS, STRATEGY_TO_MODE,
)
from beam_engine.methods.bs_kernel.calibrate import calibrate
from beam_engine.methods.bs_kernel.cost_model import (
    Coefficients, Strategy, T_LARGE_CHOICES, T_SMALL,
    cost_per_beam_batch, cost_shared_batch, cost_dec_tail_batch,
    cost_decode_tail_batch, pick_strategy_batch,
)


# Cells to diagnose. Both showed +22%/+24% regret on the fp8-KV autotune.
TARGET_CELLS = [
    (16, 8192, 8),
    (64, 8192, 8),
]


def _print_cell_diagnosis(K, L_p, B, cell_times, coeff, num_kv_heads, head_dim, dtype_bytes, suffix_len):
    print(f"\n{'='*72}\nCELL  K={K}  L_p={L_p}  B={B}\n{'='*72}")
    w = _make_workload(K, L_p, suffix_len, num_kv_heads, head_dim, dtype_bytes)
    ws = [w] * B

    # Model costs per strategy. Strategy → (modeled_us, label, optional_breakdown_dict)
    model_costs: list[tuple[Strategy, float, str]] = []

    # PER_BEAM
    pb = cost_per_beam_batch(ws, coeff)
    model_costs.append((Strategy.PER_BEAM, pb, "per_beam"))

    # SHARED_*L_*POOL for depth=2, pool∈{1,2}, t_large∈{64,128}
    for depth in (2, 3):
        for pool in (1, 2):
            for t_large in T_LARGE_CHOICES:
                try:
                    cs = cost_shared_batch(ws, coeff, depth=depth, pool_count=pool, t_large=t_large)
                    name = f"shared_d{depth}_p{pool}_t{t_large}"
                    s = Strategy[f"SHARED_{depth}L_{pool}POOL"]
                    model_costs.append((s, cs, name))
                except AssertionError:
                    pass

    # SHARED_*L_DEC_TAIL
    for depth in (2, 3):
        try:
            cs = cost_dec_tail_batch(ws, coeff, depth=depth)
            s = Strategy[f"SHARED_{depth}L_DEC_TAIL"]
            model_costs.append((s, cs, f"dec_tail_d{depth}"))
        except AssertionError:
            pass

    # Picker's actual choice (uses DEFAULT_STRATEGIES filter)
    picked = pick_strategy_batch(ws, coeff)
    print(f"\nPicker says: {picked.strategy.name}  "
          f"(t_large={picked.t_large}, depth={picked.depth}, pool={picked.pool_count})")

    # Print measured times for every probe mode
    print(f"\nMeasured times (median decode_step_ms over {B} prompts × K={K}):")
    print(f"  {'mode':<10} {'measured_ms':>12}")
    for mode, t in sorted(cell_times.items(), key=lambda kv: kv[1]):
        marker = "  ← ORACLE" if t == min(cell_times.values()) else ""
        print(f"  {mode:<10} {t:>12.3f}{marker}")

    # Print modeled costs sorted
    print(f"\nModeled costs (cost_model predicted µs per decode step):")
    print(f"  {'strategy':<32} {'modeled_us':>11}")
    for s, c, name in sorted(model_costs, key=lambda x: x[1]):
        picker_mark = "  ← picker" if s == picked.strategy else ""
        print(f"  {name:<32} {c:>11.2f}{picker_mark}")

    # Cost breakdown for the two competing strategies on B=8 cells:
    #   shared_d2_p1_t64 (picker)  vs  dec_tail_d2 (oracle for B=8)
    print(f"\nCost breakdown — competing strategies:")
    _breakdown_shared(ws, coeff, depth=2, pool=1, t_large=64, num_kv_heads=num_kv_heads, head_dim=head_dim)
    _breakdown_dec_tail(ws, coeff, depth=2)


def _breakdown_shared(workloads, c, *, depth, pool, t_large, num_kv_heads, head_dim):
    from beam_engine.methods.bs_kernel.cost_model import (
        _per_prompt_levels, _level_tiles_and_bytes, _level_bw_us, _ceil_div, _cascade_merge_us,
    )
    print(f"  SHARED_{depth}L_{pool}POOL t={t_large}:")
    total_large = 0
    total_small = 0
    bw_us = 0.0
    for w in workloads:
        levels = _per_prompt_levels(w, depth)
        large, small, _b = _level_tiles_and_bytes(levels, pool_count=pool, t_large=t_large)
        total_large += large
        total_small += small
        bw_us += _level_bw_us(levels, c)
    waves_large = max(1, _ceil_div(total_large, c.num_sms)) if total_large > 0 else 0
    waves_small = max(1, _ceil_div(total_small, c.num_sms)) if total_small > 0 else 0
    compute_us = waves_large * c.per_tile_us[t_large] + waves_small * c.per_tile_us[T_SMALL]
    merge_us = _cascade_merge_us(depth, workloads, c)
    print(f"    bw_us       = {bw_us:>9.2f}")
    print(f"    total_large_tiles = {total_large}  waves_large = {waves_large}  per_tile_us[{t_large}] = {c.per_tile_us[t_large]:.3f}")
    print(f"    total_small_tiles = {total_small}  waves_small = {waves_small}  per_tile_us[{T_SMALL}] = {c.per_tile_us[T_SMALL]:.3f}")
    print(f"    compute_us  = {compute_us:>9.2f}")
    print(f"    merge_us    = {merge_us:>9.2f}")
    print(f"    TOTAL       = {bw_us+compute_us+merge_us:>9.2f}")


def _breakdown_dec_tail(workloads, c, *, depth):
    from beam_engine.methods.bs_kernel.cost_model import (
        _per_prompt_levels_no_tail, _level_tiles_and_bytes, _level_bw_us,
        _ceil_div, _cascade_merge_us,
    )
    print(f"  SHARED_{depth}L_DEC_TAIL:")
    total_large = 0
    bw_us = 0.0
    for w in workloads:
        levels = _per_prompt_levels_no_tail(w, depth)
        large, _small, _b = _level_tiles_and_bytes(levels, pool_count=1, t_large=64)
        total_large += large
        bw_us += _level_bw_us(levels, c)
    waves_large = max(1, _ceil_div(total_large, c.num_sms)) if total_large > 0 else 0
    prefix_us = bw_us + waves_large * c.per_tile_us[64]

    # Tail
    total_kv_tokens = 0
    n_beams = 0
    bytes_per_kv = 0
    for w in workloads:
        total_kv_tokens += sum(w.suffix_lens)
        n_beams += w.K
        bytes_per_kv = w.bytes_per_kv
    waves = max(1, _ceil_div(n_beams, c.num_sms))
    mean_tail = total_kv_tokens / n_beams
    tail_bw_us = total_kv_tokens * bytes_per_kv / c.B_hbm
    tail_work_us = waves * mean_tail * c.decode_us_per_beam_kv_token
    tail_us = c.decode_launch_us + max(tail_bw_us, tail_work_us)

    merge_us = _cascade_merge_us(depth, workloads, c)
    print(f"    prefix:  bw_us={bw_us:>9.2f}  large_tiles={total_large}  waves={waves_large}  per_tile_us[64]={c.per_tile_us[64]:.3f}")
    print(f"             prefix_us = {prefix_us:.2f}")
    print(f"    tail:    n_beams={n_beams}  mean_tail={mean_tail:.1f}  waves={waves}")
    print(f"             tail_bw_us   = {tail_bw_us:>9.2f}  (= {total_kv_tokens}·{bytes_per_kv}B / B_hbm={c.B_hbm:.0f})")
    print(f"             tail_work_us = {tail_work_us:>9.2f}  (= {waves}·{mean_tail:.0f}·{c.decode_us_per_beam_kv_token:.4f})")
    print(f"             tail_us      = {tail_us:.2f}  (= launch {c.decode_launch_us:.2f} + max(bw,work))")
    print(f"    merge_us = {merge_us:.2f}")
    print(f"    TOTAL    = {prefix_us+tail_us+merge_us:.2f}")


def main():
    init_tp()
    tp_size = get_tp_world_size()
    tp_rank = get_tp_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}" if tp_size > 1 else "cuda"
    is_main = tp_rank == 0

    model_name = os.environ["BE_MODEL"]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[os.environ.get("BE_DTYPE", "bf16")]
    kv_dtype_map = {"": None, "fp16": torch.float16, "bf16": torch.bfloat16,
                    "fp8": torch.float8_e4m3fn, "fp8_e4m3": torch.float8_e4m3fn}
    kv_dtype = kv_dtype_map[os.environ.get("BE_KV_DTYPE", "")]
    if is_main:
        print(f"model={model_name} dtype={dtype} kv_dtype={kv_dtype} tp={tp_size}")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = load_model_for_causal_lm(model_name, dtype=dtype, device=device)
    config = model.config

    from dataclasses import replace as _replace
    coeffs = calibrate(device, force=False, verbose=is_main)
    defaults = Coefficients.defaults()
    coeffs = _replace(coeffs, per_tile_us=dict(defaults.per_tile_us))

    times = _measure_strategy_times(
        model, config, tok,
        coefficients=coeffs,
        grid=TARGET_CELLS,
        max_new=16,
        device=device,
        dtype=dtype,
        kv_dtype=kv_dtype,
    )

    if not is_main:
        destroy_tp()
        return

    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    _kv_for_bytes = kv_dtype if kv_dtype is not None else dtype
    dtype_bytes = torch.tensor([], dtype=_kv_for_bytes).element_size()
    suffix_len = 8

    print(f"\nCoefficients (per-tile / decode / merge):")
    print(f"  per_tile_us[64]  = {coeffs.per_tile_us[64]:.4f}")
    print(f"  per_tile_us[128] = {coeffs.per_tile_us[128]:.4f}")
    print(f"  per_tile_us[16]  = {coeffs.per_tile_us[16]:.4f}")
    print(f"  decode_launch_us = {coeffs.decode_launch_us:.4f}")
    print(f"  decode_us_per_beam_kv_token = {coeffs.decode_us_per_beam_kv_token:.4f}")
    print(f"  merge_launch_us  = {coeffs.merge_launch_us:.4f}")
    print(f"  merge_bw_us_per_row = {coeffs.merge_bw_us_per_row:.4f}")
    print(f"  B_hbm = {coeffs.B_hbm:.0f}  num_sms = {coeffs.num_sms}")

    for (K, L_p, B) in TARGET_CELLS:
        cell_times = times.get((K, L_p, B), {})
        if not cell_times:
            print(f"\n(no measurements for K={K} L_p={L_p} B={B})")
            continue
        _print_cell_diagnosis(
            K, L_p, B, cell_times, coeffs,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            dtype_bytes=dtype_bytes, suffix_len=suffix_len,
        )

    destroy_tp()


if __name__ == "__main__":
    main()

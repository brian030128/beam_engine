"""Offline (no-GPU) sweep of the DEC_TAIL switch-margin (Lever 3).

Loads the H100-calibrated coefficient cache directly, builds the
self_consistency workload (1B: K=16, B=1, L_p=32768, bf16) at a range of tail
lengths T, and reports — per margin ε — the tail length at which the picker
flips SHARED_2L_1POOL → SHARED_2L_DEC_TAIL. Goal: find ε that pushes the flip
from the measured-too-early ~1909 to the empirical per-step crossover ~4000.

    uv run python scripts/paper-exp/sweep_dec_tail_margin.py
"""

from __future__ import annotations

import json
from pathlib import Path

from beam_engine.methods.bs_kernel.calibrate import _from_payload
from beam_engine.methods.bs_kernel.cost_model import (
    Strategy, WorkloadShape, cost_shared_batch, pick_strategy_batch,
)

CACHE = Path.home() / ".cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json"

# 1B self_consistency cell.
K, L_P = 16, 32768
NUM_KV_HEADS, HEAD_DIM, NUM_QO_HEADS = 8, 64, 32
DTYPE_BYTES = 2  # bf16
BYTES_PER_KV = 2 * NUM_KV_HEADS * HEAD_DIM * DTYPE_BYTES  # k+v
KV_DTYPE = "bf16"
AVAIL = {Strategy.SHARED_2L_1POOL, Strategy.SHARED_2L_DEC_TAIL}


def _workload(tail: int) -> WorkloadShape:
    return WorkloadShape(
        K=K, L_p=L_P, suffix_lens=[tail] * K,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        bytes_per_kv=BYTES_PER_KV, num_qo_heads=NUM_QO_HEADS,
        kv_dtype=KV_DTYPE,
    )


def _costs(c, tail: int) -> tuple[float, float, str]:
    """Return (best 1pool cost, best dec_tail cost, picked strategy) at tail."""
    pick = pick_strategy_batch([_workload(tail)], c, available_strategies=AVAIL)
    one = min(v for k, v in pick.debug.items()
              if k.startswith("shared_d2_p1_t"))
    dt = min(v for k, v in pick.debug.items()
             if k.startswith("shared_d2_dec_tail_t"))
    return one, dt, pick.strategy.value


def _flip_tail(c, margin: float) -> int | None:
    """Smallest tail where the picker chooses DEC_TAIL (margin applied)."""
    c.dec_tail_switch_margin = margin
    lo, hi = 1, 12000
    for tail in range(lo, hi + 1, 50):
        _, _, s = _costs(c, tail)
        if "dec_tail" in s:
            return tail
    return None


def main() -> None:
    c = _from_payload(json.loads(CACHE.read_text()))
    c.dec_tail_switch_margin = 0.0
    print(f"loaded {CACHE.name}: num_sms={c.num_sms} B_hbm={c.B_hbm:.0f} "
          f"decode_launch_us={c.decode_launch_us:.2f} merge_launch_us={c.merge_launch_us:.2f}")

    print("\n--- cost curves (margin=0), µs --- (zoom on the cliff 1700-2300)")
    print(f"{'tail':>6} {'cost_1pool':>11} {'cost_dt':>11} {'1pool/dt':>9} {'pick':>20}")
    for tail in (500, 1000, 1700, 1800, 1900, 1950, 2000, 2100, 2300,
                 3000, 4000, 5000, 8000, 10000):
        one, dt, s = _costs(c, tail)
        print(f"{tail:>6} {one:>11.2f} {dt:>11.2f} {one / dt:>9.4f} {s:>20}")

    print("\n--- cost_1pool term breakdown across the cliff (t_large=64) ---")
    print(f"{'tail':>6} {'bw':>8} {'compute':>8} {'thrash':>7} {'waves':>5} "
          f"{'pipe?':>5} {'work':>8} {'total':>8}")
    for tail in (1800, 1900, 1940, 1950, 1960, 2000, 2200, 3000, 4000):
        tr: dict = {}
        cost_shared_batch([_workload(tail)], c, depth=2, pool_count=1,
                          t_large=64, trace=tr)
        print(f"{tail:>6} {tr['bw_us']:>8.1f} {tr['compute_us']:>8.1f} "
              f"{tr['thrash_us']:>7.1f} {tr['total_waves']:>5d} "
              f"{str(tr['well_pipelined']):>5} {tr['work_us']:>8.1f} {tr['total']:>8.1f}")

    print("\n--- flip tail (1pool→dec_tail) vs margin ε ---")
    print(f"{'margin':>8} {'flip_tail':>10}")
    for m in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.69, 0.70, 0.80):
        ft = _flip_tail(c, m)
        print(f"{m:>8.2f} {('never (>12000)' if ft is None else ft):>14}")

    # Direct solve: ε needed to flip exactly at T=4000 is cost_1pool/cost_dt - 1.
    c.dec_tail_switch_margin = 0.0
    one4, dt4, _ = _costs(c, 4000)
    print(f"\nat tail=4000: cost_1pool/cost_dt = {one4 / dt4:.4f} "
          f"-> ε≈{one4 / dt4 - 1:.3f} flips right at 4000")


if __name__ == "__main__":
    main()

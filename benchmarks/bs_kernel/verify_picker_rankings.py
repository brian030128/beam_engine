"""Compare picker rankings under the refactored CTA-granularity model
against the OLD elemental-sub-tile model on the cascade-thin-Q probe
shapes (and a few more representative shapes).

Calls ``pick_strategy_batch`` with the H100-calibrated coefficients and
prints the chosen Strategy + the cost breakdown. Useful for spotting
rank flips between NEW and OLD.

Usage:
    uv run python benchmarks/bs_kernel/verify_picker_rankings.py
"""

from __future__ import annotations
import json
from pathlib import Path

from beam_engine.methods.bs_kernel.cost_model import (
    Coefficients, WorkloadShape,
    cost_shared_batch, cost_dec_tail_batch, cost_decode_tail_batch,
    cost_per_beam_batch, pick_strategy_batch,
)
from beam_engine.methods.bs_kernel.calibrate import _from_payload


def main():
    cache = Path.home() / ".cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json"
    c = _from_payload(json.loads(cache.read_text()))
    print(f"[load] H100 coefficients from {cache.name}")
    print(f"       per_tile_per_kv_us = {c.per_tile_per_kv_us}")
    print()

    # Same probe shapes as verify_cta_granularity.py, plus larger L_p
    # closer to the 70B C3 mispick region.
    shapes = [
        # (K, L_p, L_tail, B)
        (16,   8192,  128, 1),
        (32,   8192,  128, 1),
        (32,  32768,  128, 1),
        (64,   8192,  128, 1),
        (64,  32768,  128, 1),
        # Closer to 70B C3 actual: K=32, L_p~86K, L_tail~150
        (32,  86000,  150, 1),
        # Batched (B=8) variants
        (16,   8192,  128, 8),
        (32,   8192,  128, 8),
    ]
    num_kv_heads = 4    # match the probe
    num_qo_heads = 64   # match the probe
    head_dim = 128
    bytes_per_kv = 2 * num_kv_heads * head_dim * 2

    print(f"{'K':>3} {'L_p':>6} {'L_tail':>6} {'B':>3} | "
          f"{'pick (NEW)':>16} | "
          f"{'C_fused 2L':>10} {'C_dt 2L':>10} {'C_paged':>10}")
    print("-" * 85)

    for K, L_p, L_tail, B in shapes:
        w = WorkloadShape(
            K=K, L_p=L_p, suffix_lens=[L_tail] * K,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            bytes_per_kv=bytes_per_kv,
            num_qo_heads=num_qo_heads,
        )
        wl = [w] * B
        result = pick_strategy_batch(wl, c)

        # Itemize key candidates for transparency.
        c_fused_2l_1p_128 = cost_shared_batch(wl, c, depth=2, pool_count=1, t_large=128)
        c_dt_2l = cost_dec_tail_batch(wl, c, depth=2)
        c_paged = cost_per_beam_batch(wl, c)

        pick_label = f"{result.strategy.name}"
        if hasattr(result, "t_large"):
            pick_label += f"/T={result.t_large}"
        print(f"{K:>3} {L_p:>6} {L_tail:>6} {B:>3} | "
              f"{pick_label:>16} | "
              f"{c_fused_2l_1p_128:>10.2f} {c_dt_2l:>10.2f} {c_paged:>10.2f}")


if __name__ == "__main__":
    main()

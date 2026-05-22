"""Probe whether the picker has a 1POOL→DT crossover as suffix_len grows.

If the user's hypothesis is right ("1POOL wins at small tail, DT at large"),
we should see the picker flip from 1POOL to DT as suffix grows across a
single decode loop.

Sweeps suffix_len from 1 to 512 on the 70B C3 cell and on a representative
1POOL-winning batched cell (B4 8B mfs K=32 B=4).
"""
from __future__ import annotations
import json
from pathlib import Path
from beam_engine.methods.bs_kernel.cost_model import (
    Coefficients, WorkloadShape, pick_strategy_batch,
    cost_shared_batch, cost_dec_tail_batch,
)
from beam_engine.methods.bs_kernel.calibrate import _from_payload


def sweep(label, *, K, B, L_p, num_qo, num_kv, bpkv, mn=256):
    cache = Path.home() / ".cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json"
    c = _from_payload(json.loads(cache.read_text()))
    print(f"\n=== {label}  (K={K}, B={B}, L_p={L_p}, mn={mn}) ===")
    print(f"{'suffix':>6} | {'C_1POOL':>10} {'C_DT':>10} | {'picked':>22} | win_margin")
    print("-" * 80)
    for suffix in (1, 16, 32, 64, 96, 128, 160, 192, 224, 256, 384, 512):
        w = WorkloadShape(
            K=K, L_p=L_p, suffix_lens=[suffix] * K,
            num_kv_heads=num_kv, head_dim=128,
            bytes_per_kv=bpkv, num_qo_heads=num_qo,
        )
        wl = [w] * B
        c_pool = cost_shared_batch(wl, c, depth=2, pool_count=1, t_large=128)
        c_dt   = cost_dec_tail_batch(wl, c, depth=2, t_large=128)
        result = pick_strategy_batch(wl, c)
        margin = abs(c_pool - c_dt) / min(c_pool, c_dt) * 100
        winner = "1POOL" if c_pool < c_dt else "DT"
        print(f"{suffix:>6} | {c_pool:>10.2f} {c_dt:>10.2f} | {result.strategy.name:>22} | {winner} by {margin:.1f}%")


if __name__ == "__main__":
    # 70B C3: oracle says DT by 19%
    sweep("70B C3 (mdq K=32 B=1)", K=32, B=1, L_p=86000, num_qo=32, num_kv=4,
          bpkv=2 * 4 * 128 * 1)
    # 8B B4: oracle says 1POOL by 5%
    sweep("8B B4 (mfs K=32 B=4)", K=32, B=4, L_p=6500, num_qo=32, num_kv=8,
          bpkv=2 * 8 * 128 * 2)
    # 1B A5: oracle says 1POOL by 6%
    sweep("1B A5 (mls K=32 B=4)", K=32, B=4, L_p=4000, num_qo=32, num_kv=8,
          bpkv=2 * 8 * 128 * 2)

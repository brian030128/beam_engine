"""Score the picker against the empirical 1POOL-vs-DT oracle measured
on the picker_demo grid (A2-A5, B2-B5, C3-C6).

For each cell:
  * Reconstruct (K, B, num_qo_heads, num_kv_heads, bpkv, L_p_hint,
    L_tail_hint) from the scenario name + model config.
  * Run pick_strategy_batch.
  * Compare to the empirical winner.

Outputs a correct/total tally and per-cell pick/oracle line.
"""

from __future__ import annotations
import json
from pathlib import Path

from beam_engine.methods.bs_kernel.cost_model import (
    Coefficients, WorkloadShape, pick_strategy_batch, Strategy,
)
from beam_engine.methods.bs_kernel.calibrate import _from_payload


# (cell, K, B, mn, L_p_hint, L_tail_hint, num_qo, num_kv, bpkv, oracle)
# bpkv = bytes_per_kv: 2(K+V) × num_kv × head_dim × dtype_bytes
# 1B-fp8: bf16 KV → dtype_bytes=2;  bpkv = 2·8·64·2 = 2048
# 8B-fp8: bf16 KV → dtype_bytes=2;  num_kv=8, head_dim=128 → bpkv = 4096
# 70B-fp8: fp8 KV → dtype_bytes=1; TP=2 num_kv=4, head_dim=128 → bpkv = 1024
# L_p_hint / L_tail_hint are rough scenario averages (true values
# depend on the dataset; this is for the picker prediction sanity-check
# not a precise reproduction).
CELLS = [
    # 1B (Llama-3.2-1B-FP8, bf16 KV, num_qo=32, num_kv=8, hd=64)
    ("A2_mcr_K4_B1",      4,  1,  32,  500,   16, 32,  8, 2048, "1POOL"),
    ("A3_mdq_K32_B1",    32,  1, 256, 40000, 128, 32,  8, 2048, "1POOL"),
    ("A4_mfs_K16_B8",    16,  8, 256,  6500,  80, 32,  8, 2048, "1POOL"),
    ("A5_mls_K32_B4",    32,  4, 256,  4000,  80, 32,  8, 2048, "1POOL"),
    # 8B (Llama-3.1-8B-FP8, bf16 KV, num_qo=32, num_kv=8, hd=128)
    ("B2_mcr_K4_B1",      4,  1,  32,  500,   16, 32,  8, 4096, "1POOL"),
    ("B3_mdq_K64_B1",    64,  1, 256, 40000, 128, 32,  8, 4096, "DT"),
    ("B4_mfs_K32_B4",    32,  4, 256,  6500,  80, 32,  8, 4096, "1POOL"),
    ("B5_mls_K64_B4",    64,  4, 256,  4000,  80, 32,  8, 4096, "1POOL"),
    # 70B (Llama-3-70B-FP8, fp8 KV, TP=2: num_qo=32, num_kv=4, hd=128)
    ("C3_mdq_K32_B1",    32,  1, 256, 86000, 150, 32,  4, 1024, "DT"),
    ("C4_mfs_K16_B4",    16,  4, 256,  6500,  80, 32,  4, 1024, "1POOL"),
    ("C5_mls_K32_B4",    32,  4, 256,  4000,  80, 32,  4, 1024, "1POOL"),
    ("C6_mcr_K64_B4",    64,  4, 256,   500,  80, 32,  4, 1024, "DT"),
]


def main():
    cache = Path.home() / ".cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json"
    c = _from_payload(json.loads(cache.read_text()))

    def is_1pool(s: Strategy) -> bool:
        return "1POOL" in s.name or "DEC_TAIL" not in s.name
    def is_dt(s: Strategy) -> bool:
        return "DEC_TAIL" in s.name

    print(f"{'cell':<22} {'oracle':>7} {'picked':>22} {'match':>6}")
    print("-" * 60)
    correct = 0
    total = 0
    for (cell, K, B, mn, Lp, Ltail, nqo, nkv, bpkv, oracle) in CELLS:
        w = WorkloadShape(
            K=K, L_p=Lp, suffix_lens=[Ltail] * K,
            num_kv_heads=nkv, head_dim=128,
            bytes_per_kv=bpkv, num_qo_heads=nqo,
        )
        result = pick_strategy_batch([w] * B, c)
        picked = result.strategy.name
        picked_class = "DT" if is_dt(result.strategy) else "1POOL"
        match = "✓" if picked_class == oracle else "✗"
        if picked_class == oracle:
            correct += 1
        total += 1
        print(f"{cell:<22} {oracle:>7} {picked:>22} {match:>6}")
    print(f"\n score: {correct}/{total} correct")


if __name__ == "__main__":
    main()

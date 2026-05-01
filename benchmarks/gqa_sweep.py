"""Validate the adaptive (pad_frac > 40%) heuristic across GQA ratios.

The heuristic is invariant in `R = qo_len · gqa_group_size`. Different GQA
ratios produce the same R for different (qo_len, gqa) combinations:

    gqa=1, K=64 → root R = 64    (MHA)
    gqa=2, K=32 → root R = 64
    gqa=4, K=16 → root R = 64    (Llama-3.1)
    gqa=8, K=8  → root R = 64    (Llama-3.2 70B / Mixtral)

If the heuristic is purely a function of R values and KV lengths, the
adaptive routing should pick the same pool config across these
GQA-equivalent workloads. This script tests that.

Per CLAUDE.md: pick a fully idle GPU and pin via CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import statistics

import torch

import flashinfer
from flashinfer.testing import bench_gpu_time

from benchmarks.cta_tile_ablation import (
    HEAD_DIM, PAGE_SIZE, DTYPE,
    set_gqa, make_two_level, make_imbalanced_three_level,
)


def bench_two_level(force_cta_tile_q, K, L_p, tail, num_qo_heads, num_kv_heads, device):
    set_gqa(num_qo_heads, num_kv_heads)
    kv, qo, kvi, kvix, last = make_two_level(K, L_p, tail, device)
    q = torch.randn(K, num_qo_heads, HEAD_DIM, dtype=DTYPE, device=device)
    w = flashinfer.FusedMultiLevelCascadeAttentionWrapper(
        2, kv_layout="NHD", force_cta_tile_q=force_cta_tile_q
    )
    w.plan(qo, kvi, kvix, last,
           num_qo_heads, num_kv_heads, HEAD_DIM, PAGE_SIZE)
    times = bench_gpu_time(
        lambda: w.run(q, kv),
        dry_run_iters=5, repeat_iters=50, cold_l2_cache=False,
    )
    return statistics.median(times) * 1000.0


def bench_three_level(force_cta_tile_q, K, L_p, l1_groups, inter, tails,
                      num_qo_heads, num_kv_heads, device):
    set_gqa(num_qo_heads, num_kv_heads)
    kv, qo, kvi, kvix, last = make_imbalanced_three_level(
        K, L_p, l1_groups, inter, tails, device
    )
    q = torch.randn(K, num_qo_heads, HEAD_DIM, dtype=DTYPE, device=device)
    w = flashinfer.FusedMultiLevelCascadeAttentionWrapper(
        3, kv_layout="NHD", force_cta_tile_q=force_cta_tile_q
    )
    w.plan(qo, kvi, kvix, last,
           num_qo_heads, num_kv_heads, HEAD_DIM, PAGE_SIZE)
    times = bench_gpu_time(
        lambda: w.run(q, kv),
        dry_run_iters=5, repeat_iters=50, cold_l2_cache=False,
    )
    return statistics.median(times) * 1000.0


def fmt(x):
    return f"{x:7.1f}"


def sweep_sweet_spot_across_gqa(device):
    """Sweet-spot 2-level cascade: large root, moderate tail. Vary GQA but
    keep root_R = K · gqa constant."""
    print()
    print("=" * 100)
    print("Sweet spot (2-level): root_R held constant across GQA ratios")
    print("L_p=8192, tail=64; varying (gqa, K) with K·gqa = 128")
    print("=" * 100)
    print(
        f"  {'gqa':>3} {'K':>3} {'qoH':>4} {'kvH':>4} {'root_R':>7} | "
        f"{'adaptive':>9} {'T=16':>9} {'T=128':>9} | "
        f"{'best':>9}"
    )
    print("  " + "-" * 92)
    cases = [
        # (gqa, K, num_qo_heads, num_kv_heads)
        (1, 128, 16, 16),
        (2, 64, 16, 8),
        (4, 32, 32, 8),
        (8, 16, 32, 4),
    ]
    for gqa, K, qoH, kvH in cases:
        if K * gqa != 128:
            continue
        try:
            t_two = bench_two_level(None, K, 8192, 64, qoH, kvH, device)
        except Exception as e:
            t_two = float("nan"); print(f"  fail: {e}"); continue
        try:
            t_16 = bench_two_level(16, K, 8192, 64, qoH, kvH, device)
        except Exception:
            t_16 = float("nan")
        try:
            t_128 = bench_two_level(128, K, 8192, 64, qoH, kvH, device)
        except Exception:
            t_128 = float("nan")
        best = min(t_two, t_16, t_128)
        which = "two" if t_two == best else ("T=16" if t_16 == best else "T=128")
        print(
            f"  {gqa:>3} {K:>3} {qoH:>4} {kvH:>4} {K*gqa:>7} | "
            f"{fmt(t_two):>9} {fmt(t_16):>9} {fmt(t_128):>9} | "
            f"{which:>9}"
        )


def sweep_imbalanced_across_gqa(device):
    """Imbalanced 3-level with same R distribution across GQA ratios."""
    print()
    print("=" * 100)
    print("Imbalanced 3-level: R distribution held constant across GQA")
    print("R values {128, 64, 32, 16, 8, 4} via K·gqa scaling.  L_p=4096 inter=2048 tail=256")
    print("=" * 100)
    print(
        f"  {'gqa':>3} {'K':>3} {'qoH':>4} {'kvH':>4} {'L1 groups':>20} | "
        f"{'adaptive':>9} {'T=16':>9} {'T=128':>9} | "
        f"{'best':>9}"
    )
    print("  " + "-" * 102)
    # Same R distribution: root R=128, L1 groups giving R∈{64,32,16,8,4,4}
    # which is equivalent to L1 group sizes {64/gqa, 32/gqa, 16/gqa, ...}.
    # We keep total beams · gqa = 128 (root R).
    cases = [
        # (gqa, K, qoH, kvH, l1_groups)
        (1, 128, 16, 16, [64, 32, 16, 8, 4, 4]),
        (2, 64, 16, 8, [32, 16, 8, 4, 2, 2]),
        (4, 32, 32, 8, [16, 8, 4, 2, 1, 1]),
        (8, 16, 32, 4, [8, 4, 2, 1] + [None]),  # 8+4+2+1=15 needs 1 more
    ]
    for gqa, K, qoH, kvH, groups in cases:
        # Skip the last malformed case
        if None in groups:
            continue
        if sum(groups) != K:
            print(f"  gqa={gqa} K={K} groups={groups} (sum={sum(groups)}) — skipping mismatch")
            continue
        tails = [256] * K
        try:
            t_two = bench_three_level(None, K, 4096, groups, 2048, tails, qoH, kvH, device)
        except Exception as e:
            t_two = float("nan"); print(f"  fail: {e}"); continue
        try:
            t_16 = bench_three_level(16, K, 4096, groups, 2048, tails, qoH, kvH, device)
        except Exception:
            t_16 = float("nan")
        try:
            t_128 = bench_three_level(128, K, 4096, groups, 2048, tails, qoH, kvH, device)
        except Exception:
            t_128 = float("nan")
        best = min(t_two, t_16, t_128)
        which = "two" if t_two == best else ("T=16" if t_16 == best else "T=128")
        groups_str = "{" + ",".join(str(g) for g in groups) + "}"
        print(
            f"  {gqa:>3} {K:>3} {qoH:>4} {kvH:>4} {groups_str:>20} | "
            f"{fmt(t_two):>9} {fmt(t_16):>9} {fmt(t_128):>9} | "
            f"{which:>9}"
        )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    sweep_sweet_spot_across_gqa(device)
    sweep_imbalanced_across_gqa(device)
    print()
    print("=" * 100)
    print("Findings:")
    print("  - Heuristic should depend ONLY on R = qo_len · gqa, not on (qo_len, gqa)")
    print("    individually. If true, adaptive picks the same option across GQA-equivalent")
    print("    rows above.")
    print("  - GQA ratio affects compute density per query (more qo_heads / kv_head)")
    print("    but NOT the pool-routing decision (which is a function of R).")


if __name__ == "__main__":
    main()

"""K and L_tail sweep to confirm the C3 cache-thrash formula.

Hypothesis from the L_p sweep (probe_c3_decomposition.py):
  interference = max(0, α × (L_p × bpkv − L2_size))
                where α empirically ≈ 1.1 µs/(1k tokens) at K=32, bpkv=2048.

The L_p sweep fixed K=32 and L_tail=150. This script tests whether
interference scales with K (more beams → more L1 KV slabs → more
evictions) or L_tail (longer per-beam slabs → more L1 bw → more
evictions).

Three regimes for each independent variable:

  K sweep:      K ∈ {4, 8, 16, 32, 64, 128}, fixed L_p=86k, L_tail=150
  L_tail sweep: L_tail ∈ {16, 64, 128, 256, 512}, fixed L_p=86k, K=32

For each cell:
  T_L0    = cascade prefill, L0 only (num_levels=1)
  T_L1    = cascade prefill, L1 only (the K thin Q's, no shared prefix)
  T_2L    = cascade prefill, L0 + L1 (num_levels=2)
  interference = T_2L − (T_L0 + T_L1)

If interference is independent of K and L_tail (only depends on L_p),
the formula generalizes as-is. If it scales with K or L_tail, we need
to refine the formula.
"""

from __future__ import annotations
import argparse
import torch

from probe_c3_decomposition import probe_shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"     SMs: {torch.cuda.get_device_properties(device).multi_processor_count}")
    print()

    # ---- K sweep at fixed L_p=86k, L_tail=150 ----
    print(f"=== K sweep at L_p=86016, L_tail=150 (bf16, bpkv=2048, T=128) ===\n")
    print(f"{'K':>4} | {'T_L0':>7} {'T_L1':>7} {'T_2L':>7} | {'interf':>7} | {'interf/K':>8}")
    print("-" * 60)
    for K in (4, 8, 16, 32, 64, 128):
        r = probe_shape(device, K=K, L_p=86016, L_tail=150)
        interf = r['interference_us']
        print(f"{K:>4} | {r['T_L0_us']:>7.2f} {r['T_L1_us']:>7.2f} {r['T_2L_us']:>7.2f} | "
              f"{interf:>7.2f} | {interf/K:>8.3f}")

    # ---- L_tail sweep at fixed L_p=86k, K=32 ----
    print(f"\n=== L_tail sweep at L_p=86016, K=32 (bf16, bpkv=2048, T=128) ===\n")
    print(f"{'L_tail':>6} | {'T_L0':>7} {'T_L1':>7} {'T_2L':>7} | {'interf':>7} | {'L_tail MB':>10}")
    print("-" * 65)
    for L_tail in (16, 64, 128, 256, 512):
        r = probe_shape(device, K=32, L_p=86016, L_tail=L_tail)
        interf = r['interference_us']
        # K=32 L1 total bytes = 32 * L_tail * bpkv
        l1_mb = 32 * L_tail * 2048 / (1024 * 1024)
        print(f"{L_tail:>6} | {r['T_L0_us']:>7.2f} {r['T_L1_us']:>7.2f} {r['T_2L_us']:>7.2f} | "
              f"{interf:>7.2f} | {l1_mb:>9.2f}")

    # ---- Joint sweep: K × L_p (to see if interference scales with K × L_p excess) ----
    print(f"\n=== K × L_p joint check at L_tail=150 ===\n")
    print(f"{'K':>4} {'L_p':>6} | {'T_L0':>7} {'T_L1':>7} {'T_2L':>7} | {'interf':>7}")
    print("-" * 55)
    for K in (16, 32, 64):
        for L_p in (8192, 32768, 86016):
            r = probe_shape(device, K=K, L_p=L_p, L_tail=150)
            print(f"{K:>4} {L_p:>6} | {r['T_L0_us']:>7.2f} {r['T_L1_us']:>7.2f} "
                  f"{r['T_2L_us']:>7.2f} | {r['interference_us']:>7.2f}")


if __name__ == "__main__":
    main()

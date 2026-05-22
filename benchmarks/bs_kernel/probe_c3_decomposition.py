"""Decompose the 1POOL-vs-DT gap on the 70B C3 cell.

At C3 (K=32, L_p=86k, L_tail=150, B=1, fp8 KV bpkv=1024), the cost model
predicts 1POOL should be ~1 wave of L0 work (~77 µs) because L1 is tiny
relative to L0's spare-capacity. But empirical measurements show 1POOL
pays roughly 2 waves of heavy L0 work (~154 µs of compute, ~70 µs more
than DT prefix). This probe tries to isolate the mechanism.

For each L_p ∈ {8k, 16k, 32k, 50k, 64k, 86k}:

  T_L0          = wall time of cascade prefill with L0 only (num_levels=1)
  T_L1          = wall time of cascade prefill with L1 only (the K thin Q's
                  attending K disjoint per-beam KV slabs of size L_tail —
                  no shared prefix)
  T_2L          = wall time of cascade prefill with L0 + L1 (num_levels=2,
                  what 1POOL actually does)
  T_L0 + T_L1   = "no interference" baseline (the cost if L0 and L1 didn't
                  share a kernel launch)
  Δ_interference = T_2L − (T_L0 + T_L1)

Hypotheses to discriminate:

  (a) Pure work-stealing absorption (cost model assumption):
      T_2L ≈ max(T_L0, T_L1) ≈ T_L0   (L1 hides inside L0's compute)

  (b) Independent sequential execution:
      T_2L ≈ T_L0 + T_L1   (each level pays its own time)

  (c) Super-additive (positive interference):
      T_2L > T_L0 + T_L1   (cache thrash or bw saturation: L0+L1 together
                            is slower than running them separately)

Run on the 70B-fp8 TP=2 shape (num_qo=32, num_kv=4, fp8 KV) to match the
mispick cell. fp8 chosen to keep bpkv low so we can run K=32 with long
L_p in the workspace budget.
"""

from __future__ import annotations
import argparse
import time
import torch
from flashinfer import FusedMultiLevelCascadeAttentionWrapper


def _time_us(fn, n: int = 30, warmup: int = 8, device=None) -> float:
    """Median wall time of fn over n runs (µs)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(n):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def probe_shape(
    device: torch.device,
    *,
    K: int, L_p: int, L_tail: int,
    num_kv_heads: int = 4, num_qo_heads: int = 32,
    head_dim: int = 128, page_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    force_T: int = 128,
) -> dict:
    """Time L0-only, L1-only, and L0+L1 fused for one shape."""
    n_pages_prefix = (L_p + page_size - 1) // page_size
    n_pages_tail_each = (L_tail + page_size - 1) // page_size
    total_pages = n_pages_prefix + K * n_pages_tail_each
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    kv = torch.randn(total_pages, 2, page_size, num_kv_heads, head_dim,
                     dtype=dtype, device=device)

    # L0 (root) descriptors — 1 group of K queries on L_p prefix.
    qo_indptr_L0 = torch.tensor([0, K], dtype=torch.int32, device=device)
    kvp_indptr_L0 = torch.tensor([0, n_pages_prefix], dtype=torch.int32, device=device)
    kvi_L0 = torch.arange(n_pages_prefix, dtype=torch.int32, device=device)
    kvl_L0 = torch.tensor(
        [((L_p - 1) % page_size) + 1], dtype=torch.int32, device=device,
    )

    # L1 (tail) descriptors — K groups of 1 query each on disjoint tails.
    qo_indptr_L1 = torch.arange(0, K + 1, dtype=torch.int32, device=device)
    kvp_indptr_L1 = torch.arange(
        0, (K + 1) * n_pages_tail_each, n_pages_tail_each,
        dtype=torch.int32, device=device,
    )
    kvi_L1 = (n_pages_prefix +
              torch.arange(K * n_pages_tail_each, dtype=torch.int32, device=device))
    kvl_L1 = torch.full((K,), ((L_tail - 1) % page_size) + 1,
                         dtype=torch.int32, device=device)

    # ---- num_levels=2 (L0 + L1) plan ----
    w2 = FusedMultiLevelCascadeAttentionWrapper(
        num_levels=2, float_workspace_buffer=workspace,
        kv_layout="NHD", device=device, max_levels=2,
    )
    w2.plan(
        qo_indptr_arr=[qo_indptr_L0, qo_indptr_L1],
        paged_kv_indptr_arr=[kvp_indptr_L0, kvp_indptr_L1],
        paged_kv_indices_arr=[kvi_L0, kvi_L1],
        paged_kv_last_page_len=[kvl_L0, kvl_L1],
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, page_size=page_size, causal=False,
        q_data_type=dtype, kv_data_type=dtype,
        force_cta_tile_q=force_T,
    )
    q2 = torch.randn(K + K, num_qo_heads, head_dim, dtype=dtype, device=device)
    T_2L = _time_us(lambda: w2.run(q2, kv), device=device)

    # ---- num_levels=1 with L0 only ----
    w_L0 = FusedMultiLevelCascadeAttentionWrapper(
        num_levels=1, float_workspace_buffer=workspace,
        kv_layout="NHD", device=device, max_levels=2,
    )
    w_L0.plan(
        qo_indptr_arr=[qo_indptr_L0],
        paged_kv_indptr_arr=[kvp_indptr_L0],
        paged_kv_indices_arr=[kvi_L0],
        paged_kv_last_page_len=[kvl_L0],
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, page_size=page_size, causal=False,
        q_data_type=dtype, kv_data_type=dtype,
        force_cta_tile_q=force_T,
    )
    q_L0 = torch.randn(K, num_qo_heads, head_dim, dtype=dtype, device=device)
    T_L0 = _time_us(lambda: w_L0.run(q_L0, kv), device=device)

    # ---- num_levels=1 with L1 only (K thin queries on disjoint slabs) ----
    w_L1 = FusedMultiLevelCascadeAttentionWrapper(
        num_levels=1, float_workspace_buffer=workspace,
        kv_layout="NHD", device=device, max_levels=2,
    )
    w_L1.plan(
        qo_indptr_arr=[qo_indptr_L1],
        paged_kv_indptr_arr=[kvp_indptr_L1],
        paged_kv_indices_arr=[kvi_L1],
        paged_kv_last_page_len=[kvl_L1],
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, page_size=page_size, causal=False,
        q_data_type=dtype, kv_data_type=dtype,
        force_cta_tile_q=force_T,
    )
    q_L1 = torch.randn(K, num_qo_heads, head_dim, dtype=dtype, device=device)
    T_L1 = _time_us(lambda: w_L1.run(q_L1, kv), device=device)

    return {
        "K": K, "L_p": L_p, "L_tail": L_tail, "force_T": force_T,
        "T_L0_us": T_L0,
        "T_L1_us": T_L1,
        "T_2L_us": T_2L,
        "no_interf_baseline_us": T_L0 + T_L1,
        "interference_us": T_2L - (T_L0 + T_L1),
        "L1_marginal_vs_L0_us": T_2L - T_L0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"     SMs: {torch.cuda.get_device_properties(device).multi_processor_count}")
    print()

    # K=32, L_tail=150 (C3 cell shape). Sweep L_p to see where the
    # interference kicks in.
    K, L_tail = 32, 150
    L_ps = [8192, 16384, 32768, 49152, 65536, 86016]
    # Use bf16 KV here for simplicity (probe doesn't have fp8 KV path
    # wired). bpkv = 2*4*128*2 = 2048 at bf16 — 2× the 70B-fp8 bpkv but
    # the qualitative interference behavior should be the same; only
    # the L2-fit boundary shifts.
    print(f"K={K}, L_tail={L_tail}, num_qo=32, num_kv=4, bf16 (bpkv=2048), T=128 forced\n")

    print(f"{'L_p':>7} | {'T_L0':>7} {'T_L1':>7} {'T_2L':>7} | "
          f"{'L0+L1':>7} {'2L-(L0+L1)':>10} | {'L1_marg':>8} | {'L_p MB':>7}")
    print("-" * 90)

    for L_p in L_ps:
        r = probe_shape(device, K=K, L_p=L_p, L_tail=L_tail)
        bpkv = 2 * 4 * 128 * 2  # bf16
        mb = L_p * bpkv / (1024 * 1024)
        print(f"{L_p:>7} | "
              f"{r['T_L0_us']:>7.2f} {r['T_L1_us']:>7.2f} {r['T_2L_us']:>7.2f} | "
              f"{r['no_interf_baseline_us']:>7.2f} {r['interference_us']:>10.2f} | "
              f"{r['L1_marginal_vs_L0_us']:>8.2f} | "
              f"{mb:>6.1f}")


if __name__ == "__main__":
    main()

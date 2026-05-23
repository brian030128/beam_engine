"""Microbench: BatchPrefillWithPagedKVCacheWrapper vs
FusedMultiLevelCascadeAttentionWrapper on the prefix-only problem.

DEC_TAIL's prefix-side uses BatchPrefillWithPagedKVCacheWrapper; FUSED
uses FusedMultiLevelCascadeAttentionWrapper for prefix+tail combined.
Per-layer measurement showed DEC_TAIL's prefix call costs 773 µs/call
while FUSED's entire call (prefix+tail) costs only 700 µs/call — implying
the cascade wrapper is more efficient on the prefix-style problem. This
probe isolates that question by running each wrapper on matched K × L_p
shapes.

Cells covered: K ∈ {16, 32, 64}, L_p ∈ {2048, 8192, 32768, 50000}.
"""

from __future__ import annotations

import argparse
import csv

import torch
import flashinfer


def time_us(fn, n=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000.0 / n  # µs amortised


def bench_prefill_wrapper(K, L_p, num_qo_heads, num_kv_heads, head_dim, page_size, dtype, device, n_iters=50):
    """K query rows × L_p KV tokens, attending to a shared prefix laid out
    in pages. Mimics the prefix side of DEC_TAIL: 1 prompt, K beams, all
    sharing the same prefix page list (no per-beam divergence)."""
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD",
    )
    num_pages = (L_p + page_size - 1) // page_size
    kv = torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    # one "request" of K query rows attending to the L_p prefix pages
    qo_indptr = torch.tensor([0, K], dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    last_page_len = torch.tensor(
        [max(1, L_p - (num_pages - 1) * page_size)],
        dtype=torch.int32, device=device,
    )
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=last_page_len,
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim, head_dim_vo=head_dim,
        page_size=page_size, causal=False,
        q_data_type=dtype, kv_data_type=dtype,
    )
    q = torch.randn(K, num_qo_heads, head_dim, dtype=dtype, device=device)
    return time_us(lambda: wrapper.run(q, kv), n=n_iters)


def bench_cascade_wrapper(K, L_p, num_qo_heads, num_kv_heads, head_dim, page_size, dtype, device, n_iters=50):
    """Matched workload via FusedMultiLevelCascadeAttentionWrapper. We set
    up a depth-2 cascade where level-0 is the shared prefix (L_p pages,
    K query rows) and level-1 is per-beam tails of length 0 — so the
    cascade should reduce to the prefix attention. If depth-1 isn't
    supported by the API, use a dummy depth-2 with a 1-token tail."""
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.FusedMultiLevelCascadeAttentionWrapper(
        2, workspace, kv_layout="NHD",
    )
    num_pages_pref = (L_p + page_size - 1) // page_size
    # Level 0: shared prefix — 1 group of K queries × num_pages_pref pages
    qo_indptr_l0 = torch.tensor([0, K], dtype=torch.int32, device=device)
    kv_indptr_l0 = torch.tensor([0, num_pages_pref], dtype=torch.int32, device=device)
    kv_indices_l0 = torch.arange(num_pages_pref, dtype=torch.int32, device=device)
    last_page_l0 = torch.tensor(
        [max(1, L_p - (num_pages_pref - 1) * page_size)],
        dtype=torch.int32, device=device,
    )
    # Level 1: per-beam dummy 1-token tail (K beams)
    qo_indptr_l1 = torch.arange(0, K + 1, dtype=torch.int32, device=device)
    kv_indptr_l1 = torch.arange(0, K + 1, dtype=torch.int32, device=device)
    kv_indices_l1 = torch.arange(num_pages_pref, num_pages_pref + K, dtype=torch.int32, device=device)
    last_page_l1 = torch.full((K,), 1, dtype=torch.int32, device=device)

    # Allocate KV that includes both levels
    total_pages = num_pages_pref + K
    kv = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    wrapper.plan(
        qo_indptr_arr=[qo_indptr_l0, qo_indptr_l1],
        paged_kv_indptr_arr=[kv_indptr_l0, kv_indptr_l1],
        paged_kv_indices_arr=[kv_indices_l0, kv_indices_l1],
        paged_kv_last_page_len=[last_page_l0, last_page_l1],
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size, causal=False,
        q_data_type=dtype, kv_data_type=dtype,
    )
    q = torch.randn(K, num_qo_heads, head_dim, dtype=dtype, device=device)
    return time_us(lambda: wrapper.run(q, kv), n=n_iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/bs_kernel/results/paper-exp/probe_prefix_wrappers.csv")
    ap.add_argument("--num-qo-heads", type=int, default=32)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--page-size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"num_qo_heads={args.num_qo_heads} num_kv_heads={args.num_kv_heads} head_dim={args.head_dim}\n")

    K_grid = [16, 32, 64]
    L_p_grid = [2048, 8192, 32768, 50000]

    rows = []
    for K in K_grid:
        for L_p in L_p_grid:
            try:
                t_prefill = bench_prefill_wrapper(K, L_p, args.num_qo_heads, args.num_kv_heads, args.head_dim, args.page_size, dtype, device)
            except Exception as e:
                print(f"  K={K} L_p={L_p}  BatchPrefill FAILED: {e}")
                t_prefill = float("nan")
            try:
                t_cascade = bench_cascade_wrapper(K, L_p, args.num_qo_heads, args.num_kv_heads, args.head_dim, args.page_size, dtype, device)
            except Exception as e:
                print(f"  K={K} L_p={L_p}  Cascade FAILED: {type(e).__name__}: {e}")
                t_cascade = float("nan")
            ratio = t_prefill / t_cascade if t_cascade > 0 else float("nan")
            rows.append({"K": K, "L_p": L_p, "batch_prefill_us": round(t_prefill, 2),
                         "cascade_us": round(t_cascade, 2),
                         "prefill_div_cascade": round(ratio, 3)})
            print(
                f"  K={K:>3d}  L_p={L_p:>6d}   "
                f"BatchPrefill={t_prefill:>8.2f} µs   Cascade={t_cascade:>8.2f} µs   "
                f"ratio={ratio:>5.2f}× (>1 means BatchPrefill slower)"
            )
            torch.cuda.empty_cache()
        print()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

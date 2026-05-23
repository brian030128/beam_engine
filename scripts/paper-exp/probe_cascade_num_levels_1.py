"""Verify FusedMultiLevelCascadeAttentionWrapper works at num_levels=1
after patching the >=2 constraint. Compare its output to
BatchPrefillWithPagedKVCacheWrapper for correctness, and time both
on the K=16/L_p=50K cell to confirm the cascade speedup carries over
to num_levels=1.
"""

from __future__ import annotations

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
    return s.elapsed_time(e) * 1000.0 / n


def main():
    device = torch.device("cuda")
    dtype = torch.float16
    K, L_p = 16, 50000
    num_qo_heads, num_kv_heads, head_dim = 32, 8, 128
    page_size = 16
    num_pages = (L_p + page_size - 1) // page_size

    # Same KV cache for both wrappers.
    kv = torch.randn(num_pages, 2, page_size, num_kv_heads, head_dim,
                     dtype=dtype, device=device)

    # BatchPrefill setup
    workspace_a = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    w_batch = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_a, kv_layout="NHD")
    qo_indptr = torch.tensor([0, K], dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    last_page_len = torch.tensor(
        [max(1, L_p - (num_pages - 1) * page_size)],
        dtype=torch.int32, device=device,
    )
    w_batch.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=last_page_len,
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim, head_dim_vo=head_dim,
        page_size=page_size, causal=False,
        q_data_type=dtype, kv_data_type=dtype,
    )

    # Cascade num_levels=1 setup
    workspace_b = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    w_cascade = flashinfer.FusedMultiLevelCascadeAttentionWrapper(
        1, workspace_b, kv_layout="NHD",
    )
    w_cascade.plan(
        qo_indptr_arr=[qo_indptr],
        paged_kv_indptr_arr=[paged_kv_indptr],
        paged_kv_indices_arr=[paged_kv_indices],
        paged_kv_last_page_len=[last_page_len],
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, page_size=page_size, causal=False,
        q_data_type=dtype, kv_data_type=dtype,
    )

    q = torch.randn(K, num_qo_heads, head_dim, dtype=dtype, device=device)

    # Correctness check
    out_a = w_batch.run(q, kv)
    out_b = w_cascade.run(q, kv)
    print(f"BatchPrefill output  shape={tuple(out_a.shape)}  dtype={out_a.dtype}")
    print(f"Cascade(L=1) output  shape={tuple(out_b.shape)}  dtype={out_b.dtype}")
    diff = (out_a - out_b).abs()
    print(f"max abs diff: {float(diff.max()):.6f}")
    print(f"mean abs diff: {float(diff.mean()):.6f}")
    rel = diff / (out_a.abs() + 1e-3)
    print(f"max rel diff: {float(rel.max()):.4f}")
    print(f"all close (atol=1e-2): {bool(torch.allclose(out_a, out_b, atol=1e-2))}")

    # Timing
    t_batch = time_us(lambda: w_batch.run(q, kv))
    t_cascade = time_us(lambda: w_cascade.run(q, kv))
    print(f"\nBatchPrefill: {t_batch:.2f} µs")
    print(f"Cascade(L=1): {t_cascade:.2f} µs")
    print(f"ratio: {t_batch / t_cascade:.2f}× (>1 = Cascade is faster)")


if __name__ == "__main__":
    main()

"""
Benchmark paged attention overhead vs standard flash attention.

Measures kernel-level attention latency (no model) across:
1. Flash attention (torch SDPA, forced flash backend)
2. Paged attention with 1 page (isolates paging abstraction overhead)
3. Paged attention with various page sizes (4, 8, 16, 32, 64, 128)

Uses Llama-3.1-8B attention config: 32 QO heads, 8 KV heads, head_dim=128.
"""

import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.decode import single_decode_with_kv_cache
import flashinfer.page

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Llama-3.1-8B attention config
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

SEQ_LENS = [128, 256, 512, 1024, 2048, 4096]
PAGE_SIZES = [4, 8, 16, 32, 64, 128]

WARMUP = 10
ITERS = 100


def benchmark_fn(fn, warmup=WARMUP, iters=ITERS):
    """Time a GPU function using CUDA events. Returns (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def bench_flash_attention(seq_len):
    """Benchmark torch SDPA with flash backend. Q:[1,32,S,128], K/V:[1,8,S,128]."""
    q = torch.randn(1, NUM_QO_HEADS, seq_len, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    def fn():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    return benchmark_fn(fn)


def bench_paged_attention(seq_len, page_size):
    """Benchmark FlashInfer paged prefill attention."""
    num_pages = (seq_len + page_size - 1) // page_size
    last_page_len = seq_len - (num_pages - 1) * page_size

    # KV cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_cache = torch.randn(
        num_pages, 2, page_size, NUM_KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )

    # Fill KV cache with random data via append
    k_data = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_data = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    page_indices = torch.tensor(
        [i // page_size for i in range(seq_len)], dtype=torch.int32, device=DEVICE
    )
    page_offsets = torch.tensor(
        [i % page_size for i in range(seq_len)], dtype=torch.int32, device=DEVICE
    )
    batch_indices = torch.arange(seq_len, dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.arange(seq_len + 1, dtype=torch.int32, device=DEVICE)

    flashinfer.page.append_paged_kv_cache(
        append_key=k_data,
        append_value=v_data,
        batch_indices=batch_indices,
        positions=page_offsets,
        paged_kv_cache=kv_cache,
        kv_indices=page_indices,
        kv_indptr=kv_indptr,
        kv_last_page_len=page_offsets,
        kv_layout="NHD",
    )

    # Query
    q = torch.randn(seq_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    # Wrapper metadata
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")

    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)
    paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=DEVICE)
    paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
    paged_kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=DEVICE)

    def fn():
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=NUM_QO_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim_qk=HEAD_DIM,
            page_size=page_size,
            causal=True,
            q_data_type=DTYPE,
        )
        wrapper.run(q, kv_cache)

    return benchmark_fn(fn)


def bench_flashinfer_decode(seq_len):
    """Benchmark FlashInfer single_decode_with_kv_cache. q:[NUM_QO_HEADS, HEAD_DIM], k/v:[seq_len, NUM_KV_HEADS, HEAD_DIM]."""
    q = torch.randn(NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    def fn():
        single_decode_with_kv_cache(q, k, v, kv_layout="NHD")

    return benchmark_fn(fn)


def bench_flash_decode(seq_len):
    """Benchmark torch SDPA flash backend for decode. q:[1,32,1,128], k/v:[1,8,seq_len,128]."""
    q = torch.randn(1, NUM_QO_HEADS, 1, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    def fn():
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    return benchmark_fn(fn)


def next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def main():
    print(f"Attention benchmark: {NUM_QO_HEADS} QO heads, {NUM_KV_HEADS} KV heads, "
          f"head_dim={HEAD_DIM}, dtype={DTYPE}")
    print(f"Timing: {WARMUP} warmup + {ITERS} iterations\n")

    header = f"{'seq_len':>8} | {'method':<30} | {'mean_ms':>10} | {'std_ms':>10} | {'vs_flash':>10}"
    sep = f"{'-'*8}-+-{'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}"

    # === Prefill ===
    print("=== Prefill (full sequence Q vs full sequence KV) ===\n")
    print(header)
    print(sep)

    for seq_len in SEQ_LENS:
        # 1. Flash attention baseline
        flash_mean, flash_std = bench_flash_attention(seq_len)
        print(f"{seq_len:>8} | {'flash_attn':<30} | {flash_mean:>10.4f} | {flash_std:>10.4f} | {'1.00x':>10}")

        # 2. Paged attention with 1 page (abstraction overhead)
        one_page_size = next_power_of_2(seq_len)
        mean, std = bench_paged_attention(seq_len, one_page_size)
        ratio = mean / flash_mean
        print(f"{seq_len:>8} | {f'paged_1page(ps={one_page_size})':<30} | {mean:>10.4f} | {std:>10.4f} | {ratio:>9.2f}x")

        # 3. Paged attention with various page sizes
        for ps in PAGE_SIZES:
            if ps > seq_len:
                continue
            mean, std = bench_paged_attention(seq_len, ps)
            ratio = mean / flash_mean
            print(f"{seq_len:>8} | {f'paged_ps={ps}':<30} | {mean:>10.4f} | {std:>10.4f} | {ratio:>9.2f}x")

        print(sep)

    # === Decode ===
    print("\n=== Decode (single token Q vs full sequence KV) ===\n")
    print(header)
    print(sep)

    for seq_len in SEQ_LENS:
        # 1. Flash attention decode baseline
        flash_mean, flash_std = bench_flash_decode(seq_len)
        print(f"{seq_len:>8} | {'flash_decode':<30} | {flash_mean:>10.4f} | {flash_std:>10.4f} | {'1.00x':>10}")

        # 2. FlashInfer single_decode_with_kv_cache
        mean, std = bench_flashinfer_decode(seq_len)
        ratio = mean / flash_mean
        print(f"{seq_len:>8} | {'flashinfer_decode':<30} | {mean:>10.4f} | {std:>10.4f} | {ratio:>9.2f}x")

        print(sep)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Benchmark decode attention: paged vs non-paged.

Measures kernel-level decode attention latency (1 query token vs seq_len KV):
1. Flash attention (torch SDPA) — baseline
2. FlashInfer single_decode_with_kv_cache (non-paged, batch=1 only)
3. FlashInfer BatchDecodeWithPagedKVCacheWrapper at various page sizes

Uses Llama-3.1-8B attention config: 32 QO heads, 8 KV heads, head_dim=128.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from flashinfer.decode import (
    single_decode_with_kv_cache,
    BatchDecodeWithPagedKVCacheWrapper,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Llama-3.1-8B attention config
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

BATCH_SIZES = [1, 2, 4, 8, 16]
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096]
PAGE_SIZES = [1, 4, 8, 16, 32, 64, 128]

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


def bench_flash_decode(batch_size, seq_len):
    """Benchmark torch SDPA flash backend for decode."""
    q = torch.randn(batch_size, NUM_QO_HEADS, 1, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k = torch.randn(batch_size, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v = torch.randn(batch_size, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    def fn():
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    return benchmark_fn(fn)


def bench_flashinfer_decode(seq_len):
    """Benchmark FlashInfer single_decode_with_kv_cache (batch=1 only)."""
    q = torch.randn(NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    def fn():
        single_decode_with_kv_cache(q, k, v, kv_layout="NHD")

    return benchmark_fn(fn)


def bench_paged_decode(batch_size, seq_len, page_size):
    """Benchmark FlashInfer paged decode with batch_size sequences."""
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    last_page_len = seq_len - (num_pages_per_seq - 1) * page_size
    total_pages = num_pages_per_seq * batch_size

    # Allocate a larger pool and scatter pages to simulate realistic non-contiguous access
    pool_size = 2 * total_pages
    kv_cache = torch.randn(
        pool_size, 2, page_size, NUM_KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )
    q = torch.randn(batch_size, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")

    # indptr: [0, num_pages_per_seq, 2*num_pages_per_seq, ...]
    indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=DEVICE) * num_pages_per_seq
    indices = torch.randperm(pool_size, dtype=torch.int32, device=DEVICE)[:total_pages]
    last_page_len_t = torch.full((batch_size,), last_page_len, dtype=torch.int32, device=DEVICE)

    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len_t,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=page_size,
        q_data_type=DTYPE,
    )

    def fn():
        wrapper.run(q, kv_cache)

    return benchmark_fn(fn)


def main():
    print(f"Decode attention benchmark: {NUM_QO_HEADS} QO heads, {NUM_KV_HEADS} KV heads, "
          f"head_dim={HEAD_DIM}, dtype={DTYPE}")
    print(f"Timing: {WARMUP} warmup + {ITERS} iterations\n")

    for batch_size in BATCH_SIZES:
        print(f"=== batch_size={batch_size} ===")
        header = f"{'seq_len':>8} | {'method':<30} | {'mean_ms':>10} | {'std_ms':>10} | {'vs_flash':>10}"
        sep = f"{'-'*8}-+-{'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}"

        print(header)
        print(sep)

        for seq_len in SEQ_LENS:
            # 1. Flash attention decode baseline
            flash_mean, flash_std = bench_flash_decode(batch_size, seq_len)
            print(f"{seq_len:>8} | {'flash_decode':<30} | {flash_mean:>10.4f} | {flash_std:>10.4f} | {'1.00x':>10}")

            # 2. FlashInfer single_decode (non-paged, batch=1 only)
            if batch_size == 1:
                mean, std = bench_flashinfer_decode(seq_len)
                ratio = mean / flash_mean
                print(f"{seq_len:>8} | {'flashinfer_decode':<30} | {mean:>10.4f} | {std:>10.4f} | {ratio:>9.2f}x")

            # 3. Paged decode at various page sizes
            for ps in PAGE_SIZES:
                if ps > seq_len:
                    continue
                mean, std = bench_paged_decode(batch_size, seq_len, ps)
                ratio = mean / flash_mean
                print(f"{seq_len:>8} | {f'paged_decode(ps={ps})':<30} | {mean:>10.4f} | {std:>10.4f} | {ratio:>9.2f}x")

            print(sep)

        print()

    print("Done.")


if __name__ == "__main__":
    main()

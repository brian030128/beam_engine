"""
Benchmark: cascade attention vs flat paged attention for shared-prefix decode.

Two-level cascade: level 0 = shared prefix (512 tokens), level 1 = unique suffix per seq.
Simulates decode iterations where the unique suffix grows each round.

Uses Llama-3.1-8B attention config: 32 QO heads, 8 KV heads, head_dim=128.
"""

import argparse
import math

import torch
from flashinfer.cascade import MultiLevelCascadeAttentionWrapper
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Llama-3.1-8B attention config
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

PAGE_SIZE = 16
SHARED_PREFIX_LEN = 512
UNIQUE_SUFFIX_LEN = 2  # initial
DECODE_ROUNDS = 7
NUM_SEQS_SWEEP = [4, 8, 16, 32, 64]

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


def bench_cascade_decode(num_seqs, shared_len, unique_len, page_size, use_cuda_graph):
    """Benchmark cascade attention: level 0 = shared prefix, level 1 = unique suffix."""
    num_shared_pages = math.ceil(shared_len / page_size)
    shared_last_page_len = shared_len - (num_shared_pages - 1) * page_size

    num_unique_pages = math.ceil(unique_len / page_size)
    unique_last_page_len = unique_len - (num_unique_pages - 1) * page_size

    total_pages = num_shared_pages + num_unique_pages * num_seqs
    pool_size = total_pages + 64  # some slack
    kv_cache = torch.randn(
        pool_size, 2, page_size, NUM_KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )
    q = torch.randn(num_seqs, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    # Assign pages: shared pages = [0, num_shared_pages), unique pages after that
    shared_page_indices = list(range(num_shared_pages))
    unique_page_start = num_shared_pages

    # Level 0: shared prefix — one KV "sequence" shared by all qo tokens
    l0_qo_indptr = torch.tensor([0, num_seqs], dtype=torch.int32, device=DEVICE)
    l0_kv_indptr = torch.tensor([0, num_shared_pages], dtype=torch.int32, device=DEVICE)
    l0_kv_indices = torch.tensor(shared_page_indices, dtype=torch.int32, device=DEVICE)
    l0_last_page_len = torch.tensor([shared_last_page_len], dtype=torch.int32, device=DEVICE)

    # Level 1: per-seq unique suffix
    l1_qo_indptr = torch.arange(num_seqs + 1, dtype=torch.int32, device=DEVICE)
    l1_kv_indptr = torch.arange(num_seqs + 1, dtype=torch.int32, device=DEVICE) * num_unique_pages
    l1_kv_indices_list = []
    for i in range(num_seqs):
        start = unique_page_start + i * num_unique_pages
        l1_kv_indices_list.extend(range(start, start + num_unique_pages))
    l1_kv_indices = torch.tensor(l1_kv_indices_list, dtype=torch.int32, device=DEVICE)
    l1_last_page_len = torch.full((num_seqs,), unique_last_page_len, dtype=torch.int32, device=DEVICE)

    if use_cuda_graph:
        # Pre-allocate buffer arrays for CUDA graph capture
        qo_indptr_buf_arr = [l0_qo_indptr.clone(), l1_qo_indptr.clone()]
        kv_indptr_buf_arr = [l0_kv_indptr.clone(), l1_kv_indptr.clone()]
        kv_indices_buf_arr = [l0_kv_indices.clone(), l1_kv_indices.clone()]
        kv_last_page_len_buf_arr = [l0_last_page_len.clone(), l1_last_page_len.clone()]

        wrapper = MultiLevelCascadeAttentionWrapper(
            num_levels=2,
            float_workspace_buffer=workspace_buffer,
            kv_layout="NHD",
            use_cuda_graph=True,
            qo_indptr_buf_arr=qo_indptr_buf_arr,
            paged_kv_indptr_buf_arr=kv_indptr_buf_arr,
            paged_kv_indices_buf_arr=kv_indices_buf_arr,
            paged_kv_last_page_len_buf_arr=kv_last_page_len_buf_arr,
        )
    else:
        wrapper = MultiLevelCascadeAttentionWrapper(
            num_levels=2,
            float_workspace_buffer=workspace_buffer,
            kv_layout="NHD",
        )

    wrapper.plan(
        qo_indptr_arr=[l0_qo_indptr, l1_qo_indptr],
        paged_kv_indptr_arr=[l0_kv_indptr, l1_kv_indptr],
        paged_kv_indices_arr=[l0_kv_indices, l1_kv_indices],
        paged_kv_last_page_len=[l0_last_page_len, l1_last_page_len],
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=page_size,
        q_data_type=DTYPE,
    )

    def fn():
        wrapper.run(q, kv_cache)

    return benchmark_fn(fn)


def bench_paged_decode(num_seqs, total_len, page_size, use_cuda_graph):
    """Benchmark flat paged decode: each seq has full shared+unique KV."""
    num_pages_per_seq = math.ceil(total_len / page_size)
    last_page_len = total_len - (num_pages_per_seq - 1) * page_size
    total_pages = num_pages_per_seq * num_seqs

    pool_size = total_pages + 64
    kv_cache = torch.randn(
        pool_size, 2, page_size, NUM_KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )
    q = torch.randn(num_seqs, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    if use_cuda_graph:
        indptr_buf = torch.arange(num_seqs + 1, dtype=torch.int32, device=DEVICE) * num_pages_per_seq
        indices_buf = torch.zeros(total_pages, dtype=torch.int32, device=DEVICE)
        last_page_len_buf = torch.full((num_seqs,), last_page_len, dtype=torch.int32, device=DEVICE)

        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout="NHD",
            use_cuda_graph=True,
            paged_kv_indptr_buf=indptr_buf,
            paged_kv_indices_buf=indices_buf,
            paged_kv_last_page_len_buf=last_page_len_buf,
        )
    else:
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout="NHD",
        )

    indptr = torch.arange(num_seqs + 1, dtype=torch.int32, device=DEVICE) * num_pages_per_seq
    indices = torch.arange(total_pages, dtype=torch.int32, device=DEVICE)
    last_page_len_t = torch.full((num_seqs,), last_page_len, dtype=torch.int32, device=DEVICE)

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
    parser = argparse.ArgumentParser(description="Benchmark cascade vs paged attention")
    parser.add_argument("--cuda-graph", action="store_true", help="Enable CUDA graph capture")
    args = parser.parse_args()

    use_cuda_graph = args.cuda_graph

    print(f"Cascade vs Paged attention benchmark")
    print(f"Config: {NUM_QO_HEADS} QO heads, {NUM_KV_HEADS} KV heads, head_dim={HEAD_DIM}, "
          f"page_size={PAGE_SIZE}, dtype={DTYPE}")
    print(f"Shared prefix: {SHARED_PREFIX_LEN} tokens, initial unique suffix: {UNIQUE_SUFFIX_LEN} tokens")
    print(f"Decode rounds: {DECODE_ROUNDS} (suffix grows {UNIQUE_SUFFIX_LEN} -> {UNIQUE_SUFFIX_LEN + DECODE_ROUNDS - 1})")
    print(f"CUDA graph: {use_cuda_graph}")
    print(f"Timing: {WARMUP} warmup + {ITERS} iterations\n")

    for num_seqs in NUM_SEQS_SWEEP:
        print(f"=== num_seqs={num_seqs} ===")
        header = (f"{'round':>5} | {'unique_len':>10} | {'total_len':>10} | "
                  f"{'cascade_ms':>10} | {'paged_ms':>10} | {'speedup':>8}")
        sep = (f"{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-"
               f"{'-'*10}-+-{'-'*10}-+-{'-'*8}")
        print(header)
        print(sep)

        for rnd in range(DECODE_ROUNDS):
            unique_len = UNIQUE_SUFFIX_LEN + rnd
            total_len = SHARED_PREFIX_LEN + unique_len

            cascade_mean, _ = bench_cascade_decode(
                num_seqs, SHARED_PREFIX_LEN, unique_len, PAGE_SIZE, use_cuda_graph)
            paged_mean, _ = bench_paged_decode(
                num_seqs, total_len, PAGE_SIZE, use_cuda_graph)

            speedup = paged_mean / cascade_mean
            print(f"{rnd:>5} | {unique_len:>10} | {total_len:>10} | "
                  f"{cascade_mean:>10.4f} | {paged_mean:>10.4f} | {speedup:>7.2f}x")

        print(sep)
        print()

    print("Done.")


if __name__ == "__main__":
    main()

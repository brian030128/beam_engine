"""
Single-config profile harness for the Multi-Level Max Sharing W=16 D=24
slowdown case (F/R = 1.42x). Designed for nsys/ncu capture.

Builds the workload once, runs warmup + N replays each of fused and
unfused under NVTX ranges so nsys traces show clean labelled regions.

Usage:
    CUDA_VISIBLE_DEVICES=0 nsys profile -o /tmp/cascade.nsys-rep \
      --trace cuda,nvtx \
      uv run python tests/profile_max_sharing_w16_d24.py
"""

import os
import sys
import torch
import flashinfer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from draft_tree_lib import (
    allocate_kv_cache_and_pages,
    build_max_sharing_tree,
    build_multi_level_metadata,
)


def build_fused_wrapper(meta, num_qo_heads, num_kv_heads, head_dim, page_size, dtype):
    from flashinfer.cascade import FusedMultiLevelCascadeAttentionWrapper

    num_levels = meta["num_levels"]
    kv_len_arr = []
    for lvl in range(num_levels):
        kv_indptr = meta["kv_indptr_arr"][lvl]
        last_page_len = meta["last_page_len_arr"][lvl]
        pages_per_group = kv_indptr[1:] - kv_indptr[:-1]
        kv_len = (pages_per_group - 1) * page_size + last_page_len
        kv_len = torch.where(pages_per_group > 0, kv_len, torch.zeros_like(kv_len))
        kv_len_arr.append(kv_len)

    wrapper = FusedMultiLevelCascadeAttentionWrapper(
        num_levels=num_levels,
        kv_layout="NHD",
        device="cuda",
        max_levels=64,
        force_cta_tile_q=64,  # heuristic picks this for Max Sharing
    )
    wrapper.plan(
        qo_indptr_arr=meta["qo_indptr_arr"],
        kv_indptr_arr=meta["kv_indptr_arr"],
        kv_indices_arr=meta["kv_indices_arr"],
        kv_len_arr=kv_len_arr,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    return wrapper


def build_unfused_wrapper(meta, num_qo_heads, num_kv_heads, head_dim, page_size, dtype):
    num_levels = meta["num_levels"]
    wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        num_levels,
        torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda"),
        "NHD",
    )
    wrapper.plan(
        meta["qo_indptr_arr"],
        meta["kv_indptr_arr"],
        meta["kv_indices_arr"],
        meta["last_page_len_arr"],
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True,
        q_data_type=dtype,
    )
    return wrapper


def main():
    torch.manual_seed(42)

    # Workload: Max Sharing W=16 D=24 Multi-Level (the 1.42x slowdown case)
    depth, width = 24, 16
    prompt_len = 32768
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 1
    dtype = torch.bfloat16

    edges = build_max_sharing_tree(depth, width)
    kv_data, prompt_pages, node_page_map = allocate_kv_cache_and_pages(
        edges, prompt_len, num_kv_heads, head_dim, page_size, dtype,
    )
    meta = build_multi_level_metadata(
        edges, prompt_pages, node_page_map, page_size, prompt_len,
    )
    total_q = meta["total_q"]
    print(f"Workload: total_q={total_q} num_levels={meta['num_levels']} "
          f"prompt_pages={prompt_pages} leaves={total_q}", flush=True)

    q = torch.randn(total_q, num_qo_heads, head_dim, device="cuda", dtype=dtype)

    # Build both wrappers
    fused = build_fused_wrapper(meta, num_qo_heads, num_kv_heads, head_dim, page_size, dtype)
    unfused = build_unfused_wrapper(meta, num_qo_heads, num_kv_heads, head_dim, page_size, dtype)

    # Warmup (no NVTX so warmup is excluded from analysis)
    for _ in range(10):
        fused.run(q, kv_data)
        unfused.run(q, kv_data)
    torch.cuda.synchronize()

    # Profiled region: alternating fused/unfused under NVTX so nsys
    # correlates them. Eager mode (no CUDA graph) so individual kernel
    # launches are visible in the trace.
    N_REPLAYS = 20
    for i in range(N_REPLAYS):
        torch.cuda.nvtx.range_push(f"fused_{i}")
        fused.run(q, kv_data)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"unfused_{i}")
        unfused.run(q, kv_data)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print("Profile run complete.", flush=True)


if __name__ == "__main__":
    main()

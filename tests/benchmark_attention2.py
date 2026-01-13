"""
Benchmark script for comparing:
- FlashInfer BatchDecodeWithPagedKVCacheWrapper
- FastTree attention
- FlashInfer Cascade attention

Timing-only version:
- NO torch.profiler
- NO chrome trace / output files
- Uses torch.cuda.synchronize() to ensure correctness
"""

import torch
import math
import time
import flashinfer
import flashinfer.cascade
import argparse
from dataclasses import dataclass, field
from typing import List, Optional

# Adjust python path to ensure imports work from tests/
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from beam_engine.attention.fasttree.attn_kernels import fasttree_decode
from beam_engine.attention.fasttree_adapter import prepare_fasttree_metadata_from_trie


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Mock TrieNode for FastTree Adapter
# -----------------------------------------------------------------------------
@dataclass
class TrieNode:
    tokens: List[int]
    children: List["TrieNode"] = field(default_factory=list)
    parent: Optional["TrieNode"] = None
    page_id: int = 0


# -----------------------------------------------------------------------------
# Benchmark Configuration
# -----------------------------------------------------------------------------
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PREFIX_LEN = 4096
NUM_BRANCHES = 8
BRANCH_LEN = 1
PAGE_SIZE = 16
BATCH_SIZE = 1
LEVELS = 3
DEVICE = "cuda"
WARMUP = 10
ITERATIONS = 100


def benchmark_attention(use_cuda_graph: bool = False):
    print("=" * 80)
    print("Benchmark: FlashInfer Paged vs FastTree vs Cascade")
    print(f"CUDA Graphs : {'ON' if use_cuda_graph else 'OFF'}")
    print(f"BatchSize={BATCH_SIZE}, Levels={LEVELS}")
    print(f"Prefix={PREFIX_LEN}, Branches={NUM_BRANCHES}, BranchLen={BRANCH_LEN}")
    print(f"Heads={NUM_HEADS}, KV_Heads={NUM_KV_HEADS}, HeadDim={HEAD_DIM}")
    print(f"PageSize={PAGE_SIZE}")
    print("=" * 80)

    total_requests = BATCH_SIZE * NUM_BRANCHES

    # -------------------------------------------------------------------------
    # Setup Paged KV Cache
    # -------------------------------------------------------------------------
    num_prefix_pages = math.ceil(PREFIX_LEN / PAGE_SIZE)
    num_branch_pages_per_batch = NUM_BRANCHES * (LEVELS - 1)
    pages_per_batch = num_prefix_pages + num_branch_pages_per_batch
    total_physical_pages = BATCH_SIZE * pages_per_batch

    paged_kv_cache = torch.randn(
        total_physical_pages,
        2,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.float16,
        device=DEVICE,
    )

    # -------------------------------------------------------------------------
    # FlashInfer metadata
    # -------------------------------------------------------------------------
    all_kv_page_indices = []
    kv_page_indptr = [0]
    kv_last_page_len = []

    for b in range(BATCH_SIZE):
        batch_page_offset = b * pages_per_batch
        prefix_pages = list(
            range(batch_page_offset, batch_page_offset + num_prefix_pages)
        )

        for i in range(NUM_BRANCHES):
            branch_base_page = (
                batch_page_offset + num_prefix_pages + i * (LEVELS - 1)
            )
            branch_pages = list(
                range(branch_base_page, branch_base_page + (LEVELS - 1))
            )

            req_pages = prefix_pages + branch_pages
            all_kv_page_indices.extend(req_pages)

            kv_page_indptr.append(len(all_kv_page_indices))
            kv_last_page_len.append(1)

    kv_page_indices_tensor = torch.randint(
        0,
        total_physical_pages,
        (len(all_kv_page_indices),),
        dtype=torch.int32,
        device=DEVICE,
    )
    kv_page_indptr_tensor = torch.tensor(
        kv_page_indptr, dtype=torch.int32, device=DEVICE
    )
    kv_last_page_len_tensor = torch.tensor(
        kv_last_page_len, dtype=torch.int32, device=DEVICE
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    decode_wrapper.plan(
        indptr=kv_page_indptr_tensor,
        indices=kv_page_indices_tensor,
        last_page_len=kv_last_page_len_tensor,
        num_qo_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        pos_encoding_mode="NONE",
        data_type=torch.float16,
        sm_scale=sm_scale,
    )

    q = torch.randn(
        total_requests, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device=DEVICE
    )

    # -------------------------------------------------------------------------
    # FlashInfer timing
    # -------------------------------------------------------------------------
    print("\n[FlashInfer Paged]")

    for _ in range(WARMUP):
        decode_wrapper.run(q, paged_kv_cache)

    torch.cuda.synchronize()

    if use_cuda_graph:
        g_fi = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_fi):
            decode_wrapper.run(q, paged_kv_cache)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            g_fi.replay()
    else:
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            decode_wrapper.run(q, paged_kv_cache)

    torch.cuda.synchronize()
    flashinfer_time = (time.perf_counter() - start) * 1000 / ITERATIONS
    print(f"Avg latency: {flashinfer_time:.4f} ms")

    # -------------------------------------------------------------------------
    # FastTree setup
    # -------------------------------------------------------------------------
    print("\n[FastTree]")

    super_root = TrieNode(tokens=[], page_id=-1)
    all_candidates = []

    for b in range(BATCH_SIZE):
        b_root = TrieNode(tokens=[0] * PREFIX_LEN, parent=super_root)
        super_root.children.append(b_root)

        for i in range(NUM_BRANCHES):
            curr = TrieNode(tokens=[0] * BRANCH_LEN, parent=b_root)
            b_root.children.append(curr)

            for _ in range(1, LEVELS - 1):
                child = TrieNode(tokens=[0] * BRANCH_LEN, parent=curr)
                curr.children.append(child)
                curr = child

            all_candidates.append(curr)

    total_slots = total_physical_pages * PAGE_SIZE
    fasttree_k_buffer = paged_kv_cache[:, 0].reshape(
        total_slots, NUM_KV_HEADS, HEAD_DIM
    )
    fasttree_v_buffer = paged_kv_cache[:, 1].reshape(
        total_slots, NUM_KV_HEADS, HEAD_DIM
    )

    suffix_len = (LEVELS - 1) * BRANCH_LEN
    req_to_token = torch.zeros(
        total_requests, PREFIX_LEN + suffix_len, dtype=torch.int32, device=DEVICE
    )

    for b in range(BATCH_SIZE):
        batch_page_offset = b * pages_per_batch
        prefix_start = batch_page_offset * PAGE_SIZE

        prefix_slots = torch.arange(
            prefix_start,
            prefix_start + PREFIX_LEN,
            dtype=torch.int32,
            device=DEVICE,
        )

        for i in range(NUM_BRANCHES):
            req_idx = b * NUM_BRANCHES + i
            req_to_token[req_idx, :PREFIX_LEN] = prefix_slots

            branch_base_page = (
                batch_page_offset + num_prefix_pages + i * (LEVELS - 1)
            )

            for l in range(LEVELS - 1):
                req_to_token[req_idx, PREFIX_LEN + l] = (
                    branch_base_page + l
                ) * PAGE_SIZE

    metadata = prepare_fasttree_metadata_from_trie(
        root=super_root,
        candidates=all_candidates,
        req_to_token=req_to_token,
        batch_size=total_requests,
        num_qo_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        device=DEVICE,
    )

    fasttree_o = torch.empty_like(q)

    for _ in range(WARMUP):
        fasttree_decode(
            q,
            fasttree_k_buffer,
            fasttree_v_buffer,
            fasttree_o,
            metadata.vnode_to_kv_entries,
            metadata.vnode_to_kv_offs,
            metadata.vnode_to_kv_lens,
            metadata.vnode_to_q_entries,
            metadata.vnode_to_q_offs,
            metadata.vnode_to_q_lens,
            metadata.req_to_vnode_entries,
            metadata.req_to_vnode_offs,
            metadata.req_to_vnode_lens,
            metadata.mid_o,
            metadata.mid_lse,
            metadata.phase_node_nums,
            metadata.phase_node_offsets,
            metadata.phase_q_tile_sizes,
            metadata.phase_kv_tile_sizes,
            sm_scale,
        )

    torch.cuda.synchronize()

    if use_cuda_graph:
        g_ft = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_ft):
            fasttree_decode(
                q,
                fasttree_k_buffer,
                fasttree_v_buffer,
                fasttree_o,
                metadata.vnode_to_kv_entries,
                metadata.vnode_to_kv_offs,
                metadata.vnode_to_kv_lens,
                metadata.vnode_to_q_entries,
                metadata.vnode_to_q_offs,
                metadata.vnode_to_q_lens,
                metadata.req_to_vnode_entries,
                metadata.req_to_vnode_offs,
                metadata.req_to_vnode_lens,
                metadata.mid_o,
                metadata.mid_lse,
                metadata.phase_node_nums,
                metadata.phase_node_offsets,
                metadata.phase_q_tile_sizes,
                metadata.phase_kv_tile_sizes,
                sm_scale,
            )

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            g_ft.replay()
    else:
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            fasttree_decode(
                q,
                fasttree_k_buffer,
                fasttree_v_buffer,
                fasttree_o,
                metadata.vnode_to_kv_entries,
                metadata.vnode_to_kv_offs,
                metadata.vnode_to_kv_lens,
                metadata.vnode_to_q_entries,
                metadata.vnode_to_q_offs,
                metadata.vnode_to_q_lens,
                metadata.req_to_vnode_entries,
                metadata.req_to_vnode_offs,
                metadata.req_to_vnode_lens,
                metadata.mid_o,
                metadata.mid_lse,
                metadata.phase_node_nums,
                metadata.phase_node_offsets,
                metadata.phase_q_tile_sizes,
                metadata.phase_kv_tile_sizes,
                sm_scale,
            )

    torch.cuda.synchronize()
    fasttree_time = (time.perf_counter() - start) * 1000 / ITERATIONS
    print(f"Avg latency: {fasttree_time:.4f} ms")

    # -------------------------------------------------------------------------
    # Cascade timing
    # -------------------------------------------------------------------------
    print("\n[Cascade Attention]")

    cascade_qo_indptr = []
    cascade_kv_indptr = []
    cascade_kv_indices = []
    cascade_kv_last_page_len = []

    l0_qo = [0]
    l0_kv = [0]
    l0_indices = []
    l0_last = []

    for b in range(BATCH_SIZE):
        l0_qo.append(l0_qo[-1] + NUM_BRANCHES)
        batch_page_offset = b * pages_per_batch
        prefix_pages = list(
            range(batch_page_offset, batch_page_offset + num_prefix_pages)
        )
        l0_indices.extend(prefix_pages)
        l0_kv.append(len(l0_indices))
        l0_last.append(
            PAGE_SIZE if PREFIX_LEN % PAGE_SIZE == 0 else PREFIX_LEN % PAGE_SIZE
        )

    cascade_qo_indptr.append(torch.tensor(l0_qo, device=DEVICE))
    cascade_kv_indptr.append(torch.tensor(l0_kv, device=DEVICE))
    cascade_kv_indices.append(torch.tensor(l0_indices, device=DEVICE))
    cascade_kv_last_page_len.append(torch.tensor(l0_last, device=DEVICE))

    for l in range(1, LEVELS):
        qo = [0]
        kv = [0]
        indices = []
        last = []

        for b in range(BATCH_SIZE):
            batch_page_offset = b * pages_per_batch
            for i in range(NUM_BRANCHES):
                qo.append(qo[-1] + 1)
                page = (
                    batch_page_offset
                    + num_prefix_pages
                    + i * (LEVELS - 1)
                    + (l - 1)
                )
                indices.append(page)
                kv.append(len(indices))
                last.append(1)

        cascade_qo_indptr.append(torch.tensor(qo, device=DEVICE))
        cascade_kv_indptr.append(torch.tensor(kv, device=DEVICE))
        cascade_kv_indices.append(torch.tensor(indices, device=DEVICE))
        cascade_kv_last_page_len.append(torch.tensor(last, device=DEVICE))

    workspace_cas = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    cascade_wrapper = flashinfer.cascade.MultiLevelCascadeAttentionWrapper(
        LEVELS, workspace_cas, "NHD"
    )
    print(        
        cascade_qo_indptr,
        cascade_kv_indptr,
        cascade_kv_indices,
        cascade_kv_last_page_len
    )
    cascade_wrapper.plan(
        cascade_qo_indptr,
        cascade_kv_indptr,
        cascade_kv_indices,
        cascade_kv_last_page_len,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
    )

    for _ in range(WARMUP):
        cascade_wrapper.run(q, paged_kv_cache)

    torch.cuda.synchronize()

    if use_cuda_graph:
        g_cas = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_cas):
            cascade_wrapper.run(q, paged_kv_cache)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            g_cas.replay()
    else:
        start = time.perf_counter()
        for _ in range(ITERATIONS):
            cascade_wrapper.run(q, paged_kv_cache)

    torch.cuda.synchronize()
    cascade_time = (time.perf_counter() - start) * 1000 / ITERATIONS
    print(f"Avg latency: {cascade_time:.4f} ms")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n========== Summary ==========")
    print(f"FlashInfer : {flashinfer_time:.4f} ms")
    print(f"FastTree   : {fasttree_time:.4f} ms")
    print(f"Cascade    : {cascade_time:.4f} ms")

    print("\nSpeedups:")
    print(f"FlashInfer / FastTree : {flashinfer_time / fasttree_time:.2f}x")
    print(f"FlashInfer / Cascade  : {flashinfer_time / cascade_time:.2f}x")
    print(f"Cascade    / FastTree : {cascade_time / fasttree_time:.2f}x")


if __name__ == "__main__":
    args = parse_args()
    benchmark_attention(use_cuda_graph=args.use_cuda_graph)

"""
Benchmark script for comparing FlashInfer BatchDecodeWithPagedKVCacheWrapper vs FastTree attention.
Profiling with torch.profiler.
"""

import torch
import math
from torch.profiler import profile, record_function, ProfilerActivity
import flashinfer
from dataclasses import dataclass, field
from typing import List, Optional

# Adjust python path to ensure imports work from tests/
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from beam_engine.attention.fasttree.attn_kernels import fasttree_decode
from beam_engine.attention.fasttree_adapter import prepare_fasttree_metadata_from_trie

# -----------------------------------------------------------------------------
# Mock TrieNode for FastTree Adapter
# -----------------------------------------------------------------------------
@dataclass
class TrieNode:
    tokens: List[int]
    children: List['TrieNode'] = field(default_factory=list)
    parent: Optional['TrieNode'] = None
    page_id: int = 0

# -----------------------------------------------------------------------------
# Benchmark Configuration
# -----------------------------------------------------------------------------
NUM_HEADS = 32
NUM_KV_HEADS = 32
HEAD_DIM = 128
PREFIX_LEN = 1024
NUM_BRANCHES = 8
BRANCH_LEN = 1
PAGE_SIZE = 16
DEVICE = "cuda"
WARMUP = 10
ITERATIONS = 100

def benchmark_attention():
    print("=" * 60)
    print(f"Benchmark: FlashInfer Paged vs FastTree")
    print(f"Config: Prefix={PREFIX_LEN}, Branches={NUM_BRANCHES}, BranchLen={BRANCH_LEN}")
    print(f"Heads={NUM_HEADS}, KV_Heads={NUM_KV_HEADS}, Dim={HEAD_DIM}")
    print(f"Page Size={PAGE_SIZE}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Setup Data Structures
    # -------------------------------------------------------------------------
    # Scenario: 8 requests. Each has history [Prefix (1000)] + [Branch (1)]
    # Prefix is SHARED.
    
    # 1. Setup Paged KV Cache (Physical Memory)
    # We need enough pages for Prefix + 8 * Branch
    # Prefix pages:
    num_prefix_pages = math.ceil(PREFIX_LEN / PAGE_SIZE)
    # Branch pages: 1 per branch (since 1 token < 16)
    num_branch_pages = NUM_BRANCHES
    
    total_physical_pages = num_prefix_pages + num_branch_pages
    
    # [max_num_pages, 2, page_size, num_kv_heads, head_dim]
    paged_kv_cache = torch.randn(
        total_physical_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, 
        dtype=torch.float16, device=DEVICE
    )
    
    # 2. Setup FlashInfer Metadata
    # Each request needs a list of page indices: [prefix_pages..., branch_page]
    # Prefix pages are 0..num_prefix_pages-1
    prefix_page_indices = list(range(num_prefix_pages))
    
    all_kv_page_indices = []
    kv_page_indptr = [0]
    kv_last_page_len = []
    
    for i in range(NUM_BRANCHES):
        # Branch page index: num_prefix_pages + i
        branch_page_idx = num_prefix_pages + i
        
        # Request page chain: Shared prefix pages + Unique branch page
        req_pages = prefix_page_indices + [branch_page_idx]
        all_kv_page_indices.extend(req_pages)
        
        kv_page_indptr.append(len(all_kv_page_indices))
        
        # Last page length calculations
        # Total tokens = 1000 + 1 = 1001
        # Last page len = 1001 % 16 or 16 if 0?
        # Actually FlashInfer treats last_page_len as length of valid data in the *last* page of the chain.
        # Total len 1001. 
        # (1000 // 16) = 62 pages. 1000 % 16 = 8.
        # So page 0..61 are full (16). Page 62 (index 62) has 8 tokens? 
        # Wait, 1000 / 16 = 62.5 -> 63 pages (0..62). 
        # 62 * 16 = 992. 1000 - 992 = 8. So prefix ends with 8 tokens in page 62.
        # BUT we append a new branch token!
        # If we append to the SAME sequence, we might fill the partial page?
        # Paged Attention usually appends new tokens to new pages if using block management, 
        # OR fills up the last block.
        # For simplicity benchmark: Let's assume the prefix fills complete pages + partial,
        # and checking implementation details usually requires the logic to handle "append".
        # 
        # Simpler approach matching typical "Fork" usage:
        # The history is immutable blocks. The new token is in a new block (or the last block if we copy).
        # Let's assume we allocated a UNIQUE page for the last token for each branch to handle divergence.
        # So: Prefix Pages (immutable shared) + 1 New Page (mutable unique per branch).
        # Last page len = 1 (since 1 new token in the new page).
        # Note: FlashInfer decode wrapper usually takes "kv_last_page_len" as valid length in the last page.
        kv_last_page_len.append(1) 

    kv_page_indices_tensor = torch.tensor(all_kv_page_indices, dtype=torch.int32, device=DEVICE)
    kv_page_indptr_tensor = torch.tensor(kv_page_indptr, dtype=torch.int32, device=DEVICE)
    kv_last_page_len_tensor = torch.tensor(kv_last_page_len, dtype=torch.int32, device=DEVICE)
    
    # 3. Setup Workspace for FlashInfer
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
        sm_scale=sm_scale
    )
    
    # Queries: [8, 32, 128]
    q = torch.randn(NUM_BRANCHES, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device=DEVICE)

    # -------------------------------------------------------------------------
    # Benchmark FlashInfer
    # -------------------------------------------------------------------------
    print("\nBenchmarking FlashInfer Paged Wrapper...")
    
    # Warmup
    for _ in range(WARMUP):
        decode_wrapper.run(q, paged_kv_cache)
    
    torch.cuda.synchronize()
    
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("flashinfer_paged_decode"):
            for _ in range(ITERATIONS):
                decode_wrapper.run(q, paged_kv_cache)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    flashinfer_avg_time = 0
    # Sum up kernel times for FlashInfer
    for evt in prof.key_averages():
        if "BatchDecodeWithPagedKVCacheKernel" in evt.key:
            # Check for cuda_time_total or device_time_total
            t = getattr(evt, "cuda_time_total", 0)
            if t == 0:
                t = getattr(evt, "device_time_total", 0)
            if t == 0 and hasattr(evt, "device_time"): # Fallback for old/new property
                t = evt.device_time
            elif t == 0 and hasattr(evt, "cuda_time"): # Deprecated fallback
                t = evt.cuda_time
            flashinfer_avg_time += t
            
    # Normalize by iterations if the profiler aggregated all of them
    # key_averages aggregates by kernel name. If we ran ITERATIONS times, 
    # and the kernel was called ITERATIONS times, cuda_time_total is the SUM.
    flashinfer_avg_time /= ITERATIONS
    print(f"FlashInfer Avg Time (from profiler): {flashinfer_avg_time/1000:.4f} ms")


    # -------------------------------------------------------------------------
    # Setup FastTree
    # -------------------------------------------------------------------------
    print("\nBenchmarking FastTree...")
    
    # ... (same setup code) ...
    
    # 1. Build Tree
    root = TrieNode(tokens=[0] * PREFIX_LEN, page_id=0) # Page IDs don't matter for raw kernel check, only for adapter
    branches = []
    for i in range(NUM_BRANCHES):
        child = TrieNode(tokens=[0] * BRANCH_LEN, parent=root, page_id=i+1)
        root.children.append(child)
        branches.append(child)
    
    # 2. Build req_to_token for FastTree
    # FastTree needs a logical view of the KV cache.
    # We can use the same paged physical layout concept.
    # req_to_token maps [request, token_pos] -> slot_index (global linear index).
    # Since FastTree kernel takes flattened K and V buffers, we need to map our paged cache to that.
    
    # Flattened buffer size = total_physical_pages * PAGE_SIZE
    total_slots = total_physical_pages * PAGE_SIZE
    fasttree_k_buffer = paged_kv_cache[:, 0].reshape(total_slots, NUM_KV_HEADS, HEAD_DIM)
    fasttree_v_buffer = paged_kv_cache[:, 1].reshape(total_slots, NUM_KV_HEADS, HEAD_DIM)
    
    # Construct req_to_token [8, 1001]
    req_to_token = torch.zeros(NUM_BRANCHES, PREFIX_LEN + BRANCH_LEN, dtype=torch.int32, device=DEVICE)
    
    # Map logic:
    # Prefix (0..999) -> Pages 0..62.
    #   Token k is at: (k // 16) * 16 + (k % 16) = k.
    #   Since our pages 0..62 are contiguous in paged_kv_cache linear memory (0..62*16),
    #   the slot index is just 'k'.
    prefix_indices = torch.arange(PREFIX_LEN, dtype=torch.int32, device=DEVICE)
    
    for i in range(NUM_BRANCHES):
        req_to_token[i, :PREFIX_LEN] = prefix_indices
        
        # Suffix token is in branch_page_idx (num_prefix_pages + i)
        # It's at offset 0 in that page (since new page).
        # Global slot index = (num_prefix_pages + i) * PAGE_SIZE + 0
        suffix_slot = (num_prefix_pages + i) * PAGE_SIZE
        req_to_token[i, PREFIX_LEN] = suffix_slot
        
    # 3. Metadata
    metadata = prepare_fasttree_metadata_from_trie(
        root=root,
        candidates=branches,
        req_to_token=req_to_token,
        batch_size=NUM_BRANCHES,
        num_qo_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        device=DEVICE
    )
    
    fasttree_o = torch.empty_like(q)
    
    # Warmup
    for _ in range(WARMUP):
        fasttree_decode(
            q=q,
            k_buffer=fasttree_k_buffer,
            v_buffer=fasttree_v_buffer,
            o=fasttree_o,
            vnode_to_kv_entries=metadata.vnode_to_kv_entries,
            vnode_to_kv_offs=metadata.vnode_to_kv_offs,
            vnode_to_kv_lens=metadata.vnode_to_kv_lens,
            vnode_to_q_entries=metadata.vnode_to_q_entries,
            vnode_to_q_offs=metadata.vnode_to_q_offs,
            vnode_to_q_lens=metadata.vnode_to_q_lens,
            req_to_vnode_entries=metadata.req_to_vnode_entries,
            req_to_vnode_offs=metadata.req_to_vnode_offs,
            req_to_vnode_lens=metadata.req_to_vnode_lens,
            mid_o=metadata.mid_o,
            mid_lse=metadata.mid_lse,
            phase_node_nums=metadata.phase_node_nums,
            phase_node_offsets=metadata.phase_node_offsets,
            phase_q_tile_sizes=metadata.phase_q_tile_sizes,
            phase_kv_tile_sizes=metadata.phase_kv_tile_sizes,
            sm_scale=sm_scale,
        )

    # Benchmark
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof_ft:
        with record_function("fasttree_decode"):
            for _ in range(ITERATIONS):
                fasttree_decode(
                    q=q,
                    k_buffer=fasttree_k_buffer,
                    v_buffer=fasttree_v_buffer,
                    o=fasttree_o,
                    vnode_to_kv_entries=metadata.vnode_to_kv_entries,
                    vnode_to_kv_offs=metadata.vnode_to_kv_offs,
                    vnode_to_kv_lens=metadata.vnode_to_kv_lens,
                    vnode_to_q_entries=metadata.vnode_to_q_entries,
                    vnode_to_q_offs=metadata.vnode_to_q_offs,
                    vnode_to_q_lens=metadata.vnode_to_q_lens,
                    req_to_vnode_entries=metadata.req_to_vnode_entries,
                    req_to_vnode_offs=metadata.req_to_vnode_offs,
                    req_to_vnode_lens=metadata.req_to_vnode_lens,
                    mid_o=metadata.mid_o,
                    mid_lse=metadata.mid_lse,
                    phase_node_nums=metadata.phase_node_nums,
                    phase_node_offsets=metadata.phase_node_offsets,
                    phase_q_tile_sizes=metadata.phase_q_tile_sizes,
                    phase_kv_tile_sizes=metadata.phase_kv_tile_sizes,
                    sm_scale=sm_scale,
                )

    print(prof_ft.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    fasttree_avg_time = 0
    # Sum up kernel times for FastTree (Stage 1 + Stage 2)
    for evt in prof_ft.key_averages():
        if "_fwd_fasttree_" in evt.key:
             # Check for cuda_time_total or device_time_total
            t = getattr(evt, "cuda_time_total", 0)
            if t == 0:
                t = getattr(evt, "device_time_total", 0)
            if t == 0 and hasattr(evt, "device_time"): # Fallback for old/new property
                t = evt.device_time
            elif t == 0 and hasattr(evt, "cuda_time"): # Deprecated fallback
                t = evt.cuda_time
            fasttree_avg_time += t

    # Normalize by iterations
    fasttree_avg_time /= ITERATIONS
    print(f"FastTree Avg Time (from profiler): {fasttree_avg_time/1000:.4f} ms")
    
    if fasttree_avg_time > 0 and flashinfer_avg_time > 0:
        print(f"Speedup: {flashinfer_avg_time / fasttree_avg_time:.2f}x")

    # -------------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------------
    print("\nVerifying Correctness...")
    
    # Run both once to ensure we have clean outputs
    flashinfer_output = decode_wrapper.run(q, paged_kv_cache)
    
    fasttree_decode(
        q=q,
        k_buffer=fasttree_k_buffer,
        v_buffer=fasttree_v_buffer,
        o=fasttree_o,
        vnode_to_kv_entries=metadata.vnode_to_kv_entries,
        vnode_to_kv_offs=metadata.vnode_to_kv_offs,
        vnode_to_kv_lens=metadata.vnode_to_kv_lens,
        vnode_to_q_entries=metadata.vnode_to_q_entries,
        vnode_to_q_offs=metadata.vnode_to_q_offs,
        vnode_to_q_lens=metadata.vnode_to_q_lens,
        req_to_vnode_entries=metadata.req_to_vnode_entries,
        req_to_vnode_offs=metadata.req_to_vnode_offs,
        req_to_vnode_lens=metadata.req_to_vnode_lens,
        mid_o=metadata.mid_o,
        mid_lse=metadata.mid_lse,
        phase_node_nums=metadata.phase_node_nums,
        phase_node_offsets=metadata.phase_node_offsets,
        phase_q_tile_sizes=metadata.phase_q_tile_sizes,
        phase_kv_tile_sizes=metadata.phase_kv_tile_sizes,
        sm_scale=sm_scale,
    )
    
    # Compare
    # Note: Outputs might have slight numerical differences due to order of operations/tile sizes
    max_diff = (flashinfer_output - fasttree_o).abs().max().item()
    print(f"Max Difference: {max_diff:.6f}")
    
    if max_diff < 1e-2: # Relaxed tolerance for float16
        print("✅ Outputs match!")
    else:
        print("❌ Outputs mismatch!")

if __name__ == "__main__":
    benchmark_attention()

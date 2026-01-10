"""
Benchmark script for comparing FlashInfer BatchDecodeWithPagedKVCacheWrapper vs FastTree attention.
Profiling with torch.profiler.
"""

import torch
import math
from torch.profiler import profile, record_function, ProfilerActivity
import flashinfer
import flashinfer.cascade
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

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)
def benchmark_attention():
    print("=" * 60)
    print(f"Benchmark: FlashInfer Paged vs FastTree")
    print(f"Config: BatchSize={BATCH_SIZE}, Levels={LEVELS}, Prefix={PREFIX_LEN}, Branches={NUM_BRANCHES}, BranchLen={BRANCH_LEN}")
    print(f"Heads={NUM_HEADS}, KV_Heads={NUM_KV_HEADS}, Dim={HEAD_DIM}")
    print(f"Page Size={PAGE_SIZE}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Setup Data Structures
    # -------------------------------------------------------------------------
    # Scenario: BATCH_SIZE independent trees. 
    # Each tree: 1 prefix (PREFIX_LEN) -> 8 branches.
    # Each branch extends for (LEVELS - 1) levels. Each level has 1 token.
    
    total_requests = BATCH_SIZE * NUM_BRANCHES
    
    # -------------------------------------------------------------------------
    # 1. Setup Paged KV Cache (Physical Memory)
    # -------------------------------------------------------------------------
    num_prefix_pages = math.ceil(PREFIX_LEN / PAGE_SIZE)
    # Each branch needs (LEVELS - 1) pages (1 page per level for simplicity/isolation)
    num_branch_pages_per_batch = NUM_BRANCHES * (LEVELS - 1)
    
    pages_per_batch = num_prefix_pages + num_branch_pages_per_batch
    total_physical_pages = BATCH_SIZE * pages_per_batch
    
    # [total_physical_pages, 2, page_size, num_kv_heads, head_dim]
    paged_kv_cache = torch.randn(
        total_physical_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, 
        dtype=torch.float16, device=DEVICE
    )
    
    # -------------------------------------------------------------------------
    # 2. Setup FlashInfer Metadata
    # -------------------------------------------------------------------------
    all_kv_page_indices = []
    kv_page_indptr = [0]
    kv_last_page_len = []
    
    # Loop over batches
    for b in range(BATCH_SIZE):
        batch_page_offset = b * pages_per_batch
        
        # Pages for this batch's prefix
        prefix_pages = list(range(batch_page_offset, batch_page_offset + num_prefix_pages))
        
        for i in range(NUM_BRANCHES):
            # Branch pages start after prefix pages
            # Offset for this branch: i * (LEVELS - 1)
            branch_base_page = batch_page_offset + num_prefix_pages + i * (LEVELS - 1)
            branch_pages = list(range(branch_base_page, branch_base_page + (LEVELS - 1)))
            
            # Request chain
            req_pages = prefix_pages + branch_pages
            all_kv_page_indices.extend(req_pages)
            
            kv_page_indptr.append(len(all_kv_page_indices))
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
    
    # Queries: [total_requests, heads, dim]
    q = torch.randn(total_requests, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device=DEVICE)

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
    prof.export_chrome_trace("trace_flashinfer.json")


    # -------------------------------------------------------------------------
    # Setup FastTree
    # -------------------------------------------------------------------------
    print("\nBenchmarking FastTree...")
    
    # ... (same setup code) ...
    
    # 1. Build Trees (Multiple Batches)
    super_root = TrieNode(tokens=[], page_id=-1)
    batch_roots = []
    all_candidates = []
    
    for b in range(BATCH_SIZE):
        # Prefix for batch b (PREFIX_LEN tokens)
        b_root = TrieNode(tokens=[0]*PREFIX_LEN, parent=super_root, page_id=b*1000)
        super_root.children.append(b_root)
        batch_roots.append(b_root)
        
        for i in range(NUM_BRANCHES):
            # Chains of length (LEVELS - 1)
            # Level 1 node
            curr_node = TrieNode(tokens=[0]*BRANCH_LEN, parent=b_root, page_id=b*10000 + i*100 + 0)
            b_root.children.append(curr_node)
            
            # Additional levels (2 to LEVELS-1)
            for l in range(1, LEVELS - 1):
                child = TrieNode(tokens=[0]*BRANCH_LEN, parent=curr_node, page_id=b*10000 + i*100 + l)
                curr_node.children.append(child)
                curr_node = child
                
            # The leaf node is the candidate
            all_candidates.append(curr_node)
    
    # 2. Build req_to_token for FastTree
    # Flattened buffer size = total_physical_pages * PAGE_SIZE
    total_slots = total_physical_pages * PAGE_SIZE
    fasttree_k_buffer = paged_kv_cache[:, 0].reshape(total_slots, NUM_KV_HEADS, HEAD_DIM)
    fasttree_v_buffer = paged_kv_cache[:, 1].reshape(total_slots, NUM_KV_HEADS, HEAD_DIM)
    
    # Construct req_to_token [total_requests, PREFIX_LEN + (LEVELS-1)*BRANCH_LEN]
    suffix_len = (LEVELS - 1) * BRANCH_LEN
    req_to_token = torch.zeros(total_requests, PREFIX_LEN + suffix_len, dtype=torch.int32, device=DEVICE)
    
    for b in range(BATCH_SIZE):
        batch_page_offset = b * pages_per_batch
        
        # Calculate start slot index for this batch's prefix
        prefix_start_slot = batch_page_offset * PAGE_SIZE
        prefix_slots = torch.arange(prefix_start_slot, prefix_start_slot + PREFIX_LEN, dtype=torch.int32, device=DEVICE)
        
        for i in range(NUM_BRANCHES):
            req_idx = b * NUM_BRANCHES + i
            
            # Map prefix
            req_to_token[req_idx, :PREFIX_LEN] = prefix_slots
            
            # Map suffix
            branch_base_page = batch_page_offset + num_prefix_pages + i * (LEVELS - 1)
            
            # Fill suffix tokens (LEVELS-1 tokens, each in a separate page at offset 0)
            for l in range(LEVELS - 1):
                page_idx = branch_base_page + l
                slot = page_idx * PAGE_SIZE
                req_to_token[req_idx, PREFIX_LEN + l] = slot
            
    # 3. Metadata
    metadata = prepare_fasttree_metadata_from_trie(
        root=super_root,
        candidates=all_candidates,
        req_to_token=req_to_token,
        batch_size=total_requests,
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
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof_ft:
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
    prof_ft.export_chrome_trace("trace_fasttree.json")
    
    if fasttree_avg_time > 0 and flashinfer_avg_time > 0:
        print(f"Speedup: {flashinfer_avg_time / fasttree_avg_time:.2f}x")

    # -------------------------------------------------------------------------
    # Setup Cascade Attention
    # -------------------------------------------------------------------------
    print("\nBenchmarking Cascade Attention...")
    
    cascade_qo_indptr = []
    cascade_kv_indptr = []
    cascade_kv_indices = []
    cascade_kv_last_page_len = []
    
    # Level 0 Construction (Prefix)
    l0_qo_indptr = [0]
    l0_kv_indptr = [0]
    l0_kv_indices = []
    l0_last_page_len = []
    
    for b in range(BATCH_SIZE):
        # Queries 0..(NUM_BRANCHES-1) map to this prefix node
        l0_qo_indptr.append(l0_qo_indptr[-1] + NUM_BRANCHES)
        
        # KV pages for node b
        batch_page_offset = b * pages_per_batch
        prefix_pages = list(range(batch_page_offset, batch_page_offset + num_prefix_pages))
        l0_kv_indices.extend(prefix_pages)
        
        l0_kv_indptr.append(len(l0_kv_indices))
        l0_last_page_len.append(PAGE_SIZE if (PREFIX_LEN % PAGE_SIZE == 0) else (PREFIX_LEN % PAGE_SIZE))

    cascade_qo_indptr.append(torch.tensor(l0_qo_indptr, dtype=torch.int32, device=DEVICE))
    cascade_kv_indptr.append(torch.tensor(l0_kv_indptr, dtype=torch.int32, device=DEVICE))
    cascade_kv_indices.append(torch.tensor(l0_kv_indices, dtype=torch.int32, device=DEVICE))
    cascade_kv_last_page_len.append(torch.tensor(l0_last_page_len, dtype=torch.int32, device=DEVICE))
    
    # Levels 1 to LEVELS-1
    for l in range(1, LEVELS):
        curr_qo_indptr = [0]
        curr_kv_indptr = [0]
        curr_kv_indices = []
        curr_last_page_len = []
        
        for b in range(BATCH_SIZE):
            batch_page_offset = b * pages_per_batch
            for i in range(NUM_BRANCHES):
                # One query per branch node at this level
                curr_qo_indptr.append(curr_qo_indptr[-1] + 1)
                
                # KV page for branch at THIS level
                # Branch pages start at batch_page_offset + num_prefix_pages
                # Branch i: offset i * (LEVELS-1)
                # Level l (1-based index 1..LEVELS-1) corresponds to index (l-1) within the branch
                branch_base_page = batch_page_offset + num_prefix_pages + i * (LEVELS - 1)
                page_idx = branch_base_page + (l - 1)
                
                curr_kv_indices.append(page_idx)
                curr_kv_indptr.append(len(curr_kv_indices))
                curr_last_page_len.append(1)
                
        cascade_qo_indptr.append(torch.tensor(curr_qo_indptr, dtype=torch.int32, device=DEVICE))
        cascade_kv_indptr.append(torch.tensor(curr_kv_indptr, dtype=torch.int32, device=DEVICE))
        cascade_kv_indices.append(torch.tensor(curr_kv_indices, dtype=torch.int32, device=DEVICE))
        cascade_kv_last_page_len.append(torch.tensor(curr_last_page_len, dtype=torch.int32, device=DEVICE))

    # Wrapper
    workspace_buffer_cas = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    cascade_wrapper = flashinfer.cascade.MultiLevelCascadeAttentionWrapper(
        LEVELS, workspace_buffer_cas, "NHD"
    )
    
    cascade_wrapper.plan(
        cascade_qo_indptr,
        cascade_kv_indptr,
        cascade_kv_indices,
        cascade_kv_last_page_len,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE
    )
    
    # Warmup
    for _ in range(WARMUP):
        cascade_wrapper.run(q, paged_kv_cache)
        
    torch.cuda.synchronize()
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof_cas:
        with record_function("run"):
            for _ in range(ITERATIONS):
                cascade_wrapper.run(q, paged_kv_cache)

    print(prof_cas.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    cascade_avg_time = 0
    # Sum up kernel times for Cascade (usually BatchDecode again, plus merge)
    for evt in prof_cas.key_averages():
        # Cascade calls BatchDecode kernel multiple times + merge kernel
        # Broaden filter to catch all FlashInfer related kernels
        if "flashinfer" in evt.key or "Decode" in evt.key or "Merge" in evt.key:
             # Check for cuda_time_total or device_time_total
            t = getattr(evt, "cuda_time_total", 0)
            if t == 0:
                t = getattr(evt, "device_time_total", 0)
            if t == 0 and hasattr(evt, "device_time"):
                t = evt.device_time
            elif t == 0 and hasattr(evt, "cuda_time"):
                t = evt.cuda_time
            cascade_avg_time += t

    cascade_avg_time /= ITERATIONS
    print(f"Cascade Avg Time (from profiler): {cascade_avg_time/1000:.4f} ms")
    prof_cas.export_chrome_trace("trace_cascade.json")
    
    if cascade_avg_time > 0:
        print(f"Speedup vs Cascade: {cascade_avg_time / fasttree_avg_time:.2f}x")

    # -------------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------------
    print("\nVerifying Correctness...")
    
    # Run all once to ensure we have clean outputs
    flashinfer_output = decode_wrapper.run(q, paged_kv_cache)
    cascade_output = cascade_wrapper.run(q, paged_kv_cache)
    
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
    max_diff_fi = (flashinfer_output - fasttree_o).abs().max().item()
    print(f"Max Difference (FlashInfer vs FastTree): {max_diff_fi:.6f}")
    
    max_diff_cas = (cascade_output - fasttree_o).abs().max().item()
    print(f"Max Difference (Cascade vs FastTree): {max_diff_cas:.6f}")
    
    if max_diff_cas < 1e-2:
        print("✅ Outputs match Cascade Attention!")
    elif max_diff_fi < 1e-2:
        print("✅ Outputs match FlashInfer Paged!")
    else:
        print("❌ Outputs mismatch!")

if __name__ == "__main__":
    benchmark_attention()

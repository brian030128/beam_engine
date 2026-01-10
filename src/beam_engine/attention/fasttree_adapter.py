"""
FastTree adapter for beam_engine.

Adapts BeamState trie structure to FastTree's expected input format.
Works directly with TrieNode without intermediate conversions.
"""

import torch
import queue
from typing import List, TYPE_CHECKING

from beam_engine.attention.fasttree.metadata import FastTreeMetadata

if TYPE_CHECKING:
    from beam_engine.beam_state import TrieNode


def prepare_fasttree_metadata_from_trie(
    root: 'TrieNode',
    candidates: List['TrieNode'],  # List of leaf nodes
    req_to_token: torch.Tensor,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str = "cuda",
) -> FastTreeMetadata:
    """
    Prepare FastTree metadata directly from TrieNode structure.
    
    Args:
        root: Root TrieNode of the tree
        candidates: List of leaf TrieNodes (one per request/candidate)
        req_to_token: Page table tensor (batch_size, max_seqlen)
        batch_size: Number of requests (beam candidates)
        num_qo_heads: Number of query/output heads
        num_kv_heads: Number of key/value heads
        head_dim: Head dimension
        device: Device to allocate tensors on
    
    Returns:
        FastTreeMetadata with populated buffers
    """
    metadata = FastTreeMetadata(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )
    
    # Cost model parameters
    alpha, beta, gamma = metadata.alpha, metadata.beta, metadata.gamma
    kv_group_num = num_qo_heads // num_kv_heads
    phase_q_tile_sizes = list(metadata.TSQs)
    phase_kv_tile_sizes = list(metadata.TSKs)
    phase_kv_split_sizes = [metadata.kv_split_sizes[0], metadata.kv_split_sizes[0]]
    
    def CpadQ(TS, N):
        return TS - ((N - 1) % TS + 1)
    
    def CpadK(TS, N):
        return max(0, TS - N)
    
    def Cmm(nQ, nK):
        phase = 0 if nQ > phase_q_tile_sizes[1] else 1
        TSQ = phase_q_tile_sizes[phase]
        TSK = phase_kv_tile_sizes[phase]
        return alpha * CpadQ(TSQ, nQ) * kv_group_num * nK + beta * CpadK(TSK, nK) * nQ * kv_group_num
    
    def SplitQCost(nQcurr, nQl, lenv, lenl):
        return Cmm(nQcurr - nQl, lenv) + Cmm(nQl, lenl + lenv)
    
    def SplitKCost(nQcurr, nQl, lenl, lenv):
        return Cmm(nQcurr, lenv) + Cmm(nQl, lenl) + gamma * nQl
    
    # Build node mapping and tree structure
    node_to_id = {}
    id_to_node = {}
    node_id = 0
    
    def assign_ids(node: 'TrieNode'):
        """Recursively assign IDs to all nodes."""
        nonlocal node_id
        if id(node) not in node_to_id:
            current_id = node_id
            node_to_id[id(node)] = current_id
            id_to_node[current_id] = node
            node_id += 1
            for child in node.children:
                assign_ids(child)
    
    assign_ids(root)
    num_nodes = len(node_to_id)
    
    # Build edges and request assignments
    edges = [[] for _ in range(num_nodes)]
    requests_per_node = [[] for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        node = id_to_node[i]
        if node.parent is not None:
            parent_id = node_to_id[id(node.parent)]
            edges[parent_id].append(i)
    
    # Assign requests to nodes
    for req_id, leaf_node in enumerate(candidates):
        node = leaf_node
        while node is not None:
            node_id_val = node_to_id[id(node)]
            if req_id not in requests_per_node[node_id_val]:
                requests_per_node[node_id_val].append(req_id)
            node = node.parent
    
    # Sort requests for consistency
    for reqs in requests_per_node:
        reqs.sort()
    
    # Compute sequence lengths
    L = [len(id_to_node[i].tokens) for i in range(num_nodes)]
    node_assignments = [0] * num_nodes
    
    # BFS traversal for cost-based splitting
    que = queue.Queue()
    que.put(0)
    while not que.empty():
        node_idx = que.get()
        nQcurr = len(requests_per_node[node_idx])
        lenv = L[node_idx]
        
        for child_idx in edges[node_idx]:
            nQl = len(requests_per_node[child_idx])
            lenl = L[child_idx]
            C0 = SplitKCost(nQcurr, nQl, lenl, lenv)
            C1 = SplitQCost(nQcurr, nQl, lenv, lenl)
            if C0 > C1:
                node_assignments[child_idx] = 1  # Merge with parent
                nQcurr -= nQl
                L[child_idx] = lenl + lenv
            else:
                node_assignments[child_idx] = 0  # Split
            que.put(child_idx)
    
    # Compute which requests each node handles after merging
    node_to_reqs = [[] for _ in range(num_nodes)]
    que = queue.Queue()
    
    # Start from leaves
    num_children = [len(edges[i]) for i in range(num_nodes)]
    virtual_children = num_children.copy()
    
    for i in range(num_nodes):
        if num_children[i] == 0:  # Leaf
            que.put(i)
            node_to_reqs[i] = requests_per_node[i].copy()
    
    while not que.empty():
        node_idx = que.get()
        node = id_to_node[node_idx]
        
        if node_assignments[node_idx] == 0 and node_idx != 0:
            parent_id = node_to_id[id(node.parent)]
            node_to_reqs[parent_id] += requests_per_node[node_idx]
        
        if node.parent is not None:
            parent_id = node_to_id[id(node.parent)]
            virtual_children[parent_id] -= 1
            if virtual_children[parent_id] == 0:
                que.put(parent_id)
    
    # Compute token offsets for each node
    node_token_offsets = [0] * num_nodes
    for i in range(1, num_nodes):
        node = id_to_node[i]
        parent_id = node_to_id[id(node.parent)]
        node_token_offsets[i] = node_token_offsets[parent_id] + len(id_to_node[parent_id].tokens)
    
    # Build vnode metadata
    vnode_to_kv_entries = []  # NEW: actual slot indices
    vnode_to_kv_offs = []
    vnode_to_kv_lens = []
    vnode_to_q_entries = []
    vnode_to_q_offs = []
    vnode_to_q_lens = []
    req_to_vnode_entries = [[] for _ in range(batch_size)]
    
    req_to_token_stride = req_to_token.stride(0)
    
    for i in range(num_nodes):
        req_num = len(node_to_reqs[i])
        if req_num == 0:
            continue
        
        # Compute merged KV length
        kv_len = L[i]
        node_idx = i
        while node_assignments[node_idx] == 1:
            node = id_to_node[node_idx]
            node_idx = node_to_id[id(node.parent)]
            # kv_len is already computed in BFS (L[i] accumulated)
        
        phase = 0 if req_num > phase_q_tile_sizes[1] else 1
        kv_split_size = phase_kv_split_sizes[phase]
        q_split_size = phase_q_tile_sizes[phase]
        
        # Get first request to find the page table row
        first_req = node_to_reqs[i][0]
        token_offset = node_token_offsets[node_idx]  # Use merged node's offset
        
        # Extract KV entries from req_to_token for this vnode
        # These are the actual slot indices in the K/V buffer
        kv_entries_for_vnode = []
        for kv_pos in range(kv_len):
            slot_idx = req_to_token[first_req, token_offset + kv_pos].item()
            kv_entries_for_vnode.append(slot_idx)
        
        kv_split_count = (kv_len - 1) // kv_split_size + 1
        q_split_count = (req_num - 1) // q_split_size + 1
        
        for kv_split_id in range(kv_split_count):
            q_offset_start = len(vnode_to_q_entries)
            for req in node_to_reqs[i]:
                vnode_to_q_entries.append(req)
            
            split_kv_off = kv_split_id * kv_split_size
            vnode_kv_len = min(split_kv_off + kv_split_size, kv_len) - split_kv_off
            
            # Add the actual KV slot indices for this split
            kv_entries_offset_start = len(vnode_to_kv_entries)
            for j in range(vnode_kv_len):
                vnode_to_kv_entries.append(kv_entries_for_vnode[split_kv_off + j])
            
            for q_split_id in range(q_split_count):
                split_q_off = q_split_id * q_split_size
                vnode_q_len = min(split_q_off + q_split_size, req_num) - split_q_off
                
                vnode_to_kv_offs.append(kv_entries_offset_start)
                vnode_to_kv_lens.append(vnode_kv_len)
                vnode_to_q_offs.append(q_offset_start + split_q_off)
                vnode_to_q_lens.append(vnode_q_len)
    
    # Build req_to_vnode mapping
    for i, req in enumerate(vnode_to_q_entries):
        req_to_vnode_entries[req].append(i)
    
    req_to_vnode_offs = []
    req_to_vnode_lens = []
    offset = 0
    for i in range(batch_size):
        req_to_vnode_offs.append(offset)
        offset += len(req_to_vnode_entries[i])
        req_to_vnode_lens.append(len(req_to_vnode_entries[i]))
    
    req_to_vnode_entries_flat = [item for sublist in req_to_vnode_entries for item in sublist]
    
    # Reorder vnodes by phase
    threshold = phase_q_tile_sizes[1]
    above_indices = [i for i, val in enumerate(vnode_to_q_lens) if val > threshold]
    below_indices = [i for i, val in enumerate(vnode_to_q_lens) if val <= threshold]
    new_order = above_indices + below_indices
    
    metadata.phase_node_nums = (len(above_indices), len(below_indices))
    metadata.phase_node_offsets = (0, len(above_indices))
    metadata.phase_q_tile_sizes = tuple(phase_q_tile_sizes)
    metadata.phase_kv_tile_sizes = tuple(phase_kv_tile_sizes)
    
    vnode_to_q_lens = [vnode_to_q_lens[i] for i in new_order]
    vnode_to_q_offs = [vnode_to_q_offs[i] for i in new_order]
    vnode_to_kv_lens = [vnode_to_kv_lens[i] for i in new_order]
    vnode_to_kv_offs = [vnode_to_kv_offs[i] for i in new_order]
    
    # Copy to GPU
    def to_gpu(preallocated, data):
        t = torch.tensor(data, dtype=torch.int32, device="cpu")
        preallocated[:len(data)].copy_(t, non_blocking=True)
    
    to_gpu(metadata.vnode_to_kv_entries, vnode_to_kv_entries)
    to_gpu(metadata.vnode_to_q_entries, vnode_to_q_entries)
    to_gpu(metadata.vnode_to_q_offs, vnode_to_q_offs)
    to_gpu(metadata.vnode_to_q_lens, vnode_to_q_lens)
    to_gpu(metadata.vnode_to_kv_offs, vnode_to_kv_offs)
    to_gpu(metadata.vnode_to_kv_lens, vnode_to_kv_lens)
    to_gpu(metadata.req_to_vnode_entries, req_to_vnode_entries_flat)
    to_gpu(metadata.req_to_vnode_offs, req_to_vnode_offs)
    to_gpu(metadata.req_to_vnode_lens, req_to_vnode_lens)
    
    return metadata

"""FastTree metadata preparation for paged KV cache."""

import torch
import queue
from typing import List

from beam_engine.kv_tree_node import KVTreeNode
from .metadata import FastTreeMetadata


def prepare_fasttree_metadata_for_paged_cache(
    tree_info: List[KVTreeNode],
    req_to_token: torch.Tensor,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str = "cuda",
) -> FastTreeMetadata:
    """
    Prepare FastTree metadata using paged KV cache (page table approach).

    The key difference from contiguous KV:
    - vnode_to_kv_offs are offsets into req_to_token (page table)
    - The kernel uses req_to_token[offset] to get actual KV slot indices
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

    # BFS traversal to compute node assignments
    node_num = len(tree_info)
    edges = [[] for _ in range(node_num)]
    for i in range(node_num):
        if tree_info[i].parent != -1:
            edges[tree_info[i].parent].append(i)

    L = [tree_info[i].seqlen for i in range(node_num)]
    node_assignments = [0] * node_num

    # Heuristic: decide split strategy per edge
    que = queue.Queue()
    que.put(0)
    while not que.empty():
        node = que.get()
        nQcurr = len(tree_info[node].requests)
        lenv = L[node]

        for child in edges[node]:
            nQl = len(tree_info[child].requests)
            lenl = L[child]
            C0 = SplitKCost(nQcurr, nQl, lenl, lenv)
            C1 = SplitQCost(nQcurr, nQl, lenv, lenl)
            if C0 > C1:
                node_assignments[child] = 1  # Merge with parent
                nQcurr -= nQl
                L[child] = lenl + lenv
            else:
                node_assignments[child] = 0  # Split
            que.put(child)

    # Compute which requests each node handles after merging
    node_to_reqs = [[] for _ in range(node_num)]
    que = queue.Queue()
    for i in range(node_num):
        if tree_info[i].num_children == 0:
            que.put(i)
            node_to_reqs[i] = tree_info[i].requests.copy()

    virtual_children = [tree_info[n].num_children for n in range(node_num)]
    while not que.empty():
        node = que.get()
        if node_assignments[node] == 0 and node != 0:
            node_to_reqs[tree_info[node].parent] += tree_info[node].requests
        virtual_children[tree_info[node].parent] -= 1
        if tree_info[node].parent >= 0 and virtual_children[tree_info[node].parent] == 0:
            que.put(tree_info[node].parent)

    # Build vnode metadata
    # Key: vnode_to_kv_offs are offsets into req_to_token (the page table)
    vnode_to_kv_offs = []
    vnode_to_kv_lens = []
    vnode_to_q_entries = []
    vnode_to_q_offs = []
    vnode_to_q_lens = []
    req_to_vnode_entries = [[] for _ in range(batch_size)]

    req_to_token_stride = req_to_token.stride(0)

    # Compute token offsets for each node (position in the sequence)
    node_token_offsets = [0] * node_num
    for i in range(1, node_num):
        parent = tree_info[i].parent
        # Token offset = parent's offset + parent's tokens
        node_token_offsets[i] = node_token_offsets[parent] + tree_info[parent].seqlen

    for i in range(node_num):
        req_num = len(node_to_reqs[i])
        if req_num == 0:
            continue

        # Compute merged KV length
        kv_len = tree_info[i].seqlen
        node = i
        while node_assignments[node] == 1:
            node = tree_info[node].parent
            kv_len += tree_info[node].seqlen

        phase = 0 if req_num > phase_q_tile_sizes[1] else 1
        kv_split_size = phase_kv_split_sizes[phase]
        q_split_size = phase_q_tile_sizes[phase]

        # Get first request to find the page table row
        first_req = node_to_reqs[i][0]
        token_offset = node_token_offsets[node]  # Use merged node's offset

        # KV offset is into the page table (req_to_token)
        kv_offset_start = req_to_token_stride * first_req + token_offset

        kv_split_count = (kv_len - 1) // kv_split_size + 1
        q_split_count = (req_num - 1) // q_split_size + 1

        for kv_split_id in range(kv_split_count):
            q_offset_start = len(vnode_to_q_entries)
            for req in node_to_reqs[i]:
                vnode_to_q_entries.append(req)

            split_kv_off = kv_split_id * kv_split_size
            vnode_kv_len = min(split_kv_off + kv_split_size, kv_len) - split_kv_off

            for q_split_id in range(q_split_count):
                split_q_off = q_split_id * q_split_size
                vnode_q_len = min(split_q_off + q_split_size, req_num) - split_q_off

                vnode_to_kv_offs.append(kv_offset_start + split_kv_off)
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

    to_gpu(metadata.vnode_to_q_entries, vnode_to_q_entries)
    to_gpu(metadata.vnode_to_q_offs, vnode_to_q_offs)
    to_gpu(metadata.vnode_to_q_lens, vnode_to_q_lens)
    to_gpu(metadata.vnode_to_kv_offs, vnode_to_kv_offs)
    to_gpu(metadata.vnode_to_kv_lens, vnode_to_kv_lens)
    to_gpu(metadata.req_to_vnode_entries, req_to_vnode_entries_flat)
    to_gpu(metadata.req_to_vnode_offs, req_to_vnode_offs)
    to_gpu(metadata.req_to_vnode_lens, req_to_vnode_lens)

    # Key Fix: Assign the flattened page table to vnode_to_kv_entries
    # The kernel uses indirect indexing: vnode_to_kv_offs points into this buffer
    metadata.vnode_to_kv_entries = req_to_token.flatten().int()

    return metadata

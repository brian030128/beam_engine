"""Baseline 3 — FastTree (MLSys'25, arXiv:2502.18030), integrated with the
shared page_driver beam-search loop.

FastTree's Triton kernel reads K/V via a flat slot index
``slot = page * page_size + offset``. With our PageTable layout
``[2, max_pages, page_size, num_kv_heads, head_dim]``, ``kv[0]`` and ``kv[1]``
are contiguous slabs that view as ``[max_pages * page_size, num_kv_heads,
head_dim]`` with no copy — exactly the flat layout the kernel expects.

The radix tree is built at PAGE level (16× less data than slot level).
Each tree node owns a contiguous run of pages; we expand to slot indices
only when emitting the kernel's ``vnode_to_kv_entries`` array. The
per-beam leaf's last page may be partial (``current_pos % page_size + 1``
tokens written so far).

Reuses the upstream FastTree heuristic (``_tree_heuristic`` from the
artifact) and Triton kernel (``fasttree_decode``) unchanged.
"""

from __future__ import annotations

import queue as _queue
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..decoding import (
    DecodeSelect,
    PrefillSelect,
    standard_decode_select,
    standard_prefill_select,
)
from ..methods.adaptive_pool import Beam
from ..models.attention import AttentionContext
from ..page_driver import (
    StepPlan,
    WrapperBundle,
    beam_search as _shared_beam_search,
)
from ..page_table import PageTable


# ---------------------------------------------------------------------------
# Pull FastTree's Triton kernel + heuristic from the artifact submodule.
# ---------------------------------------------------------------------------

_FT_DIR = (
    Path(__file__).resolve().parents[3]
    / "3rdparty"
    / "FastTree-Artifact"
    / "kernel_bench"
)
if str(_FT_DIR) not in sys.path:
    sys.path.insert(0, str(_FT_DIR))

from fasttree import (  # noqa: E402  (sys.path injected above)
    FastTreeParams,
    _tree_heuristic,
    fasttree_decode,
)
from kv_tree_simple import KVTreeNode  # noqa: E402


# ---------------------------------------------------------------------------
# Page-level radix-tree construction
# ---------------------------------------------------------------------------


def _build_radix_tree_pages_single(
    shared_prefix: list[int],
    tails: list[list[int]],
) -> tuple[list[KVTreeNode], list[list[int]]]:
    """Compress K beam page-paths into a radix tree.

    Beams share ``shared_prefix`` (by construction in our beam search,
    after split-pages CoW) and have their own ``tails[i]``. The root
    node's run = ``shared_prefix`` (aliased; never mutated) plus any
    further shared tail pages. The walker only scans the divergent
    tail portion, avoiding the O(K × prefix_len) prefix scan.

    Returns ``(tree_info, node_pages)`` where:
      * ``tree_info[i]``: KVTreeNode with ``parent``, ``num_children``,
        ``requests`` (beam IDs), and ``seqlen`` set in **pages**. The caller
        is expected to scale to tokens after the partial-last-page check.
      * ``node_pages[i]``: pages owned by node i (in KV order).
    """
    K = len(tails)
    nodes: list[KVTreeNode] = []
    node_pages: list[list[int]] = []

    # Fast path: at high K, beams typically diverge on different tokens
    # past the LCA, so the recursive walk emits exactly ``1 root + K
    # leaves`` with no further branching. Build that flat shape directly
    # to skip K+1 recursion frames and the per-node ``groups`` dict.
    # Mirrors the bs_kernel ``_adaptive_levels`` fast-path for the
    # analogous case. Disable via ``BE_FT_FASTPATH_RADIX=0`` for A/B.
    import os as _os
    _fast_enabled = _os.environ.get("BE_FT_FASTPATH_RADIX", "1") != "0"
    if _fast_enabled and K >= 2 and all(tails) and all(len(t) > 0 for t in tails):
        seen: set[int] = set()
        all_distinct = True
        for i in range(K):
            seen.add(tails[i][0])
            if len(seen) < i + 1:
                all_distinct = False
                break
        if all_distinct:
            # Root node (parent=-1), aliases shared_prefix, K children.
            root = KVTreeNode()
            root.parent = -1
            root.id = 0
            root.seqlen = len(shared_prefix)
            root.num_children = K
            root.requests = list(range(K))
            nodes.append(root)
            node_pages.append(shared_prefix)
            # K leaf nodes (one per beam), each with run = full tail.
            for i in range(K):
                leaf = KVTreeNode()
                leaf.parent = 0
                leaf.id = i + 1
                leaf.seqlen = len(tails[i])
                leaf.num_children = 0
                leaf.requests = [i]
                nodes.append(leaf)
                node_pages.append(tails[i])
            return nodes, node_pages

    def walk(beams: list[int], tail_d: int, parent_id: int) -> None:
        tail_len = len(tails[beams[0]])
        if parent_id == -1:
            # Root: scan for further-shared tail pages past the prefix.
            ext: list[int] = []
            d = 0
            while d < tail_len:
                page = tails[beams[0]][d]
                same = True
                for b in beams:
                    if tails[b][d] != page:
                        same = False
                        break
                if not same:
                    break
                ext.append(page)
                d += 1
            # Aliasing is safe within one plan_decode_step; if no tail
            # pages are shared, run is the prefix list itself (no copy).
            run: list[int] = shared_prefix if not ext else (shared_prefix + ext)
        else:
            run = []
            d = tail_d
            while d < tail_len:
                page = tails[beams[0]][d]
                same = True
                for b in beams:
                    if tails[b][d] != page:
                        same = False
                        break
                if not same:
                    break
                run.append(page)
                d += 1

        n = KVTreeNode()
        n.parent = parent_id
        n.id = len(nodes)
        n.seqlen = len(run)
        n.num_children = 0
        n.requests = list(beams)
        my_id = n.id
        nodes.append(n)
        node_pages.append(run)

        if d == tail_len:
            return  # leaf

        groups: dict[int, list[int]] = {}
        for b in beams:
            groups.setdefault(tails[b][d], []).append(b)
        for sub_beams in groups.values():
            walk(sub_beams, d, my_id)
            nodes[my_id].num_children += 1

    walk(list(range(K)), 0, -1)
    return nodes, node_pages


def _build_combined_radix_tree_pages(
    shared_prefix_per_prompt: list[list[int]],
    tails_per_beam_per_prompt: list[list[list[int]]],
    K: int,
) -> tuple[list[KVTreeNode], list[list[int]]]:
    """Combine B per-prompt page-radix subtrees under a virtual root.

    Each prompt's K beams share ``shared_prefix_per_prompt[b]`` (aliased
    in by reference) and have their own ``tails_per_beam_per_prompt[b][k]``.
    Request IDs span ``[0, B*K)``: prompt b's beams have IDs
    ``[b*K, (b+1)*K)``. Virtual root has ``seqlen=0`` and ``num_children=B``.
    """
    B = len(shared_prefix_per_prompt)
    virtual_root = KVTreeNode()
    virtual_root.parent = -1
    virtual_root.id = 0
    virtual_root.seqlen = 0
    virtual_root.num_children = B
    virtual_root.requests = []
    combined_nodes: list[KVTreeNode] = [virtual_root]
    combined_pages: list[list[int]] = [[]]

    for b in range(B):
        sub_nodes, sub_pages = _build_radix_tree_pages_single(
            shared_prefix_per_prompt[b], tails_per_beam_per_prompt[b],
        )
        offset = len(combined_nodes)
        for n in sub_nodes:
            n2 = KVTreeNode()
            n2.parent = (offset + n.parent) if n.parent != -1 else 0
            n2.id = offset + n.id
            n2.seqlen = n.seqlen
            n2.num_children = n.num_children
            n2.requests = [r + b * K for r in n.requests]
            combined_nodes.append(n2)
            # ``sub_pages[n.id]`` is already a fresh list; alias it.
            combined_pages.append(sub_pages[n.id])
    return combined_nodes, combined_pages


def _expand_pages_to_slots(
    nodes: list[KVTreeNode],
    node_pages: list[list[int]],
    page_size: int,
    leaf_partial_last_per_node: dict[int, int],
) -> list[np.ndarray]:
    """Flatten per-node page lists into slot arrays, handling partial leaf
    last pages.

    For non-leaf nodes (or non-partial leaves) all owned pages contribute
    the full ``page_size`` slots. For nodes in ``leaf_partial_last_per_node``
    the last page contributes only the given token count (typically
    ``current_pos[b] % page_size + 1``).

    Mutates each node's ``seqlen`` to the resulting slot count (in tokens).
    Returns numpy arrays so downstream metadata packing can use
    ``np.concatenate`` and feed ``torch.from_numpy`` without a Python-level
    list→tensor conversion.
    """
    arange_ps = np.arange(page_size, dtype=np.int32)
    node_slots: list[np.ndarray] = []
    for i, pages in enumerate(node_pages):
        if not pages:
            nodes[i].seqlen = 0
            node_slots.append(np.empty(0, dtype=np.int32))
            continue
        pages_arr = np.asarray(pages, dtype=np.int32)
        # slots[p, j] = pages[p]*ps + j ; flatten row-major.
        slots = (pages_arr[:, None] * page_size + arange_ps[None, :]).ravel()
        partial = leaf_partial_last_per_node.get(i)
        if partial is not None:
            slots = slots[: (len(pages) - 1) * page_size + partial]
        nodes[i].seqlen = int(slots.size)
        node_slots.append(slots)
    return node_slots


# ---------------------------------------------------------------------------
# FastTree metadata build (unchanged from the standalone driver)
# ---------------------------------------------------------------------------


@dataclass
class _FTMeta:
    vnode_to_kv_entries: torch.Tensor
    vnode_to_kv_offs: torch.Tensor
    vnode_to_kv_lens: torch.Tensor
    vnode_to_q_entries: torch.Tensor
    vnode_to_q_offs: torch.Tensor
    vnode_to_q_lens: torch.Tensor
    req_to_vnode_entries: torch.Tensor
    req_to_vnode_offs: torch.Tensor
    req_to_vnode_lens: torch.Tensor
    mid_o: torch.Tensor
    mid_lse: torch.Tensor
    phase_node_nums: list[int]
    phase_node_offsets: list[int]
    q_tile_sizes: list[int]
    kv_tile_sizes: list[int]


def _compute_parallelism(
    tree_info: list[KVTreeNode],
    node_assignments: list[int],
    Q_TILE_SIZE_PER_PHASE: list[int],
    KV_SPLIT_SIZE_PER_PHASE: list[int],
    num_kv_heads: int,
):
    """Mirror fasttree.compute_parallelism — counts CTAs per phase under the
    current assignment, and propagates merged requests up split-Q chains.
    """
    node_num = len(tree_info)
    node_to_reqs: list[list[int]] = [[] for _ in range(node_num)]
    que: _queue.Queue = _queue.Queue()
    for i in range(node_num):
        if tree_info[i].num_children == 0:
            que.put(i)
            node_to_reqs[i] = list(tree_info[i].requests)

    virtual_children = [tree_info[n].num_children for n in range(node_num)]
    while not que.empty():
        node = que.get()
        if node_assignments[node] == 0 and node != 0:
            node_to_reqs[tree_info[node].parent] += tree_info[node].requests
        if tree_info[node].parent != -1:
            virtual_children[tree_info[node].parent] -= 1
            if virtual_children[tree_info[node].parent] == 0:
                que.put(tree_info[node].parent)

    parallelisms = [0, 0]
    for i in range(node_num):
        req_num = len(node_to_reqs[i])
        if req_num == 0:
            continue
        node = i
        kv_len = tree_info[i].seqlen
        while node_assignments[node] == 1:
            node = tree_info[node].parent
            kv_len += tree_info[node].seqlen
        phase = 0 if req_num > Q_TILE_SIZE_PER_PHASE[1] else 1
        q_vnode_count = (req_num - 1) // Q_TILE_SIZE_PER_PHASE[phase] + 1
        kv_vnode_count = (kv_len - 1) // KV_SPLIT_SIZE_PER_PHASE[phase] + 1
        parallelisms[phase] += kv_vnode_count * q_vnode_count
    parallelisms = [p * num_kv_heads for p in parallelisms]
    return parallelisms, node_to_reqs


def _build_metadata(
    tree_info: list[KVTreeNode],
    node_slots: list[list[int]],
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    KV_SPLIT_SIZES: list[int],
    para_threshs1: list[int],
    para_threshs2: list[int],
    params: FastTreeParams,
    device: torch.device,
    wrappers: "_FastTreeWrappers | None" = None,
    cache_key: tuple | None = None,
) -> _FTMeta:
    # Heuristic-loop cache. ``_tree_heuristic`` + ``_compute_parallelism`` is
    # 0.5-1ms/step at B=4 K=32 multi_level_system shape, and re-runs 2-3
    # iterations to converge each step. The decisions depend on tree
    # structure (stable while no page boundary is crossed) and seqlens
    # (root constant, leaves grow by ≤1 token/step within a page). For our
    # workloads the decision is rock-stable between page boundaries, so we
    # key the cache on (n_nodes, total_pages) — that invalidates exactly at
    # page boundaries and on any structural change.
    #
    # Disable via BE_FT_PLAN_CACHE=0 for A/B.
    import os as _os
    _cache_enabled = (
        wrappers is not None
        and cache_key is not None
        and _os.environ.get("BE_FT_PLAN_CACHE", "1") != "0"
    )
    cache_hit = (
        _cache_enabled
        and wrappers._heur_key == cache_key
        and wrappers._heur_assignments is not None
    )
    # Time only the "dispatch decision" — the heuristic that selects
    # tile sizes + per-node SplitQ/SplitK assignments. Index packing
    # below (vnode chunk concat, numpy → torch H2D) is timed separately
    # by the caller. On cache hit the dispatch cost collapses to a few
    # list copies + a single dict lookup; on miss it runs the 3-iter
    # tree_heuristic + compute_parallelism convergence loop.
    import time as _heur_time
    _t_disp_start = _heur_time.perf_counter()
    if cache_hit:
        node_assignments = wrappers._heur_assignments
        Q_TILE_SIZE_PER_PHASE = list(wrappers._heur_q_tile)
        KV_SPLIT_SIZE_PER_PHASE = list(wrappers._heur_kv_split)
        node_to_reqs = wrappers._heur_node_to_reqs
        # Heuristic mutated ``params`` (set_q_tile_sizes / set_kv_tile_sizes)
        # on the original computation; restore so downstream uses see the
        # converged values.
        params.set_q_tile_sizes(list(wrappers._heur_q_tile))
        params.set_kv_tile_sizes(list(wrappers._heur_kv_tsk))
    else:
        Q_TILE_SIZE_PER_PHASE = list(params.TSQs)
        KV_SPLIT_SIZE_PER_PHASE = [KV_SPLIT_SIZES[0], KV_SPLIT_SIZES[0]]

        node_assignments: list[int] = []
        node_to_reqs: list[list[int]] = []
        for it in range(3):
            node_assignments = _tree_heuristic(tree_info, params)
            parallelisms, node_to_reqs = _compute_parallelism(
                tree_info,
                node_assignments,
                Q_TILE_SIZE_PER_PHASE,
                KV_SPLIT_SIZE_PER_PHASE,
                num_kv_heads,
            )
            if it == 0:
                done = True
                for phase in range(2):
                    if 0 < parallelisms[phase] < para_threshs1[phase]:
                        KV_SPLIT_SIZE_PER_PHASE[phase] = KV_SPLIT_SIZES[1]
                        done = False
                if done:
                    break
            elif it == 1:
                if 0 < parallelisms[0] < para_threshs2[0]:
                    Q_TILE_SIZE_PER_PHASE = [Q_TILE_SIZE_PER_PHASE[1]] * 2
                    params.set_q_tile_sizes(Q_TILE_SIZE_PER_PHASE)
                    params.set_kv_tile_sizes([params.TSKs[1]] * 2)
                elif 0 < parallelisms[1] < para_threshs2[1]:
                    Q_TILE_SIZE_PER_PHASE = [Q_TILE_SIZE_PER_PHASE[0]] * 2
                    params.set_q_tile_sizes(Q_TILE_SIZE_PER_PHASE)
                    params.set_kv_tile_sizes([params.TSKs[0]] * 2)
                else:
                    break

        if _cache_enabled:
            wrappers._heur_key = cache_key
            wrappers._heur_assignments = node_assignments
            wrappers._heur_q_tile = list(Q_TILE_SIZE_PER_PHASE)
            wrappers._heur_kv_split = list(KV_SPLIT_SIZE_PER_PHASE)
            wrappers._heur_kv_tsk = list(params.TSKs)
            wrappers._heur_node_to_reqs = node_to_reqs

    _dispatch_ms = (_heur_time.perf_counter() - _t_disp_start) * 1000.0
    if wrappers is not None:
        wrappers._last_dispatch_ms = _dispatch_ms

    Q_TILE_SIZE = Q_TILE_SIZE_PER_PHASE[0]
    # ``vnode_to_kv_chunks`` collects per-vnode KV-slot arrays as numpy
    # chunks (avoids per-step Python list extends of ~262K ints). Other
    # offsets/lens are small (~1000 entries) so plain Python lists.
    vnode_to_kv_chunks: list[np.ndarray] = []
    total_kv_entries = 0
    vnode_to_kv_offs: list[int] = []
    vnode_to_kv_lens: list[int] = []
    vnode_to_q_entries: list[int] = []
    vnode_to_q_offs: list[int] = []
    vnode_to_q_lens: list[int] = []

    node_num = len(tree_info)
    for i in range(node_num):
        req_num = len(node_to_reqs[i])
        if req_num == 0:
            continue
        phase = 0 if req_num > Q_TILE_SIZE_PER_PHASE[1] else 1
        KV_SPLIT_SIZE = KV_SPLIT_SIZE_PER_PHASE[phase]

        node = i
        chunks: list[np.ndarray] = [node_slots[node]]
        kv_len = tree_info[node].seqlen
        while node_assignments[node] == 1:
            node = tree_info[node].parent
            kv_len += tree_info[node].seqlen
            chunks.insert(0, node_slots[node])
        curr_KV_indices = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

        acc_kv_len = total_kv_entries
        vnode_to_kv_chunks.append(curr_KV_indices)
        total_kv_entries += curr_KV_indices.size

        kv_vnode_count = (kv_len - 1) // KV_SPLIT_SIZE + 1
        q_vnode_count = (req_num - 1) // Q_TILE_SIZE + 1
        for kv_vnode_id in range(kv_vnode_count):
            acc_q_len = len(vnode_to_q_entries)
            vnode_to_q_entries.extend(node_to_reqs[i])
            split_kv_off = kv_vnode_id * KV_SPLIT_SIZE
            vnode_kv_len = min(split_kv_off + KV_SPLIT_SIZE, kv_len) - split_kv_off
            for q_vnode_id in range(q_vnode_count):
                split_q_off = q_vnode_id * Q_TILE_SIZE
                vnode_q_len = min(split_q_off + Q_TILE_SIZE, req_num) - split_q_off
                vnode_to_kv_offs.append(acc_kv_len + split_kv_off)
                vnode_to_kv_lens.append(vnode_kv_len)
                vnode_to_q_offs.append(acc_q_len + split_q_off)
                vnode_to_q_lens.append(vnode_q_len)

    # req → vnode reduction metadata (stage 2). Vectorised with numpy:
    # the Python list-of-lists + flatten was 0.1-0.3ms at typical sizes.
    q_arr = np.asarray(vnode_to_q_entries, dtype=np.int32)
    n_vnodes = q_arr.size
    if n_vnodes > 0:
        counts = np.bincount(q_arr, minlength=batch_size)
        # ``argsort(stable)`` reorders vnode indices so that all vnodes for
        # request r appear contiguously — exactly the kernel's expected
        # ``req_to_vnode_entries`` layout.
        req_to_vnode_entries_arr = np.argsort(q_arr, kind="stable").astype(np.int32)
        req_to_vnode_lens_arr = counts.astype(np.int32)
        req_to_vnode_offs_arr = np.empty(batch_size, dtype=np.int32)
        req_to_vnode_offs_arr[0] = 0
        np.cumsum(counts[:-1], out=req_to_vnode_offs_arr[1:])
    else:
        req_to_vnode_entries_arr = np.empty(0, dtype=np.int32)
        req_to_vnode_lens_arr = np.zeros(batch_size, dtype=np.int32)
        req_to_vnode_offs_arr = np.zeros(batch_size, dtype=np.int32)

    # Phase split + reorder vnodes by Q-tile size (matches fasttree.py).
    threshold = Q_TILE_SIZE_PER_PHASE[1]
    q_lens_np = np.asarray(vnode_to_q_lens, dtype=np.int32)
    q_offs_np = np.asarray(vnode_to_q_offs, dtype=np.int32)
    kv_lens_np = np.asarray(vnode_to_kv_lens, dtype=np.int32)
    kv_offs_np = np.asarray(vnode_to_kv_offs, dtype=np.int32)
    above_mask = q_lens_np > threshold
    above_idx = np.nonzero(above_mask)[0]
    below_idx = np.nonzero(~above_mask)[0]
    new_order = np.concatenate([above_idx, below_idx]).astype(np.int32) if n_vnodes > 0 else np.empty(0, dtype=np.int32)
    phase_node_nums = [int(above_idx.size), int(below_idx.size)]
    phase_node_offsets = [0, int(above_idx.size)]
    q_lens_np = q_lens_np[new_order]
    q_offs_np = q_offs_np[new_order]
    kv_lens_np = kv_lens_np[new_order]
    kv_offs_np = kv_offs_np[new_order]
    q_entries_np = q_arr  # already int32; q_entries is not phase-reordered

    # Consolidate KV chunks into a single int32 numpy array + padding.
    pad = np.full(64, -1, dtype=np.int32)
    if vnode_to_kv_chunks:
        kv_entries_np = np.concatenate(vnode_to_kv_chunks + [pad])
    else:
        kv_entries_np = pad.copy()
    req_to_vnode_entries_np = np.concatenate(
        [req_to_vnode_entries_arr, np.full(64, -1, dtype=np.int32)]
    )

    # Single H2D for all int32 metadata. Building one contiguous host buffer
    # and copying it in a single ``.to(device)`` is ~9x fewer kernel launches
    # than the prior tensor-per-array path; on multi_level_system this was
    # ~0.5ms of the per-step CPU plan time.
    arrays = (
        kv_entries_np,                # 0
        kv_offs_np,                   # 1
        kv_lens_np,                   # 2
        q_entries_np,                 # 3
        q_offs_np,                    # 4
        q_lens_np,                    # 5
        req_to_vnode_entries_np,      # 6
        req_to_vnode_offs_arr,        # 7
        req_to_vnode_lens_arr,        # 8
    )
    sizes = [a.size for a in arrays]
    starts = [0]
    for s in sizes:
        starts.append(starts[-1] + s)
    big = np.concatenate(arrays) if starts[-1] > 0 else np.empty(0, dtype=np.int32)
    big_t = torch.from_numpy(big).to(device, non_blocking=True)
    def _slice(i):
        return big_t[starts[i]:starts[i + 1]]

    n_q_entries = max(1, len(vnode_to_q_entries))
    # Reuse mid_o, mid_lse across steps (allocated lazily / grown when needed).
    if wrappers is not None:
        cur_mid_o = wrappers._mid_o_buf
        cur_mid_lse = wrappers._mid_lse_buf
        need = (
            cur_mid_o is None
            or cur_mid_o.shape[0] < n_q_entries
            or cur_mid_o.shape[1] != num_qo_heads
            or cur_mid_o.shape[2] != head_dim
        )
        if need:
            wrappers._mid_o_buf = torch.empty(
                (n_q_entries, num_qo_heads, head_dim),
                dtype=torch.float32, device=device,
            )
            wrappers._mid_lse_buf = torch.empty(
                (n_q_entries, num_qo_heads),
                dtype=torch.float32, device=device,
            )
        mid_o = wrappers._mid_o_buf[:n_q_entries]
        mid_lse = wrappers._mid_lse_buf[:n_q_entries]
    else:
        mid_o = torch.empty(
            (n_q_entries, num_qo_heads, head_dim), dtype=torch.float32, device=device,
        )
        mid_lse = torch.empty(
            (n_q_entries, num_qo_heads), dtype=torch.float32, device=device,
        )

    return _FTMeta(
        vnode_to_kv_entries=_slice(0),
        vnode_to_kv_offs=_slice(1),
        vnode_to_kv_lens=_slice(2),
        vnode_to_q_entries=_slice(3),
        vnode_to_q_offs=_slice(4),
        vnode_to_q_lens=_slice(5),
        req_to_vnode_entries=_slice(6),
        req_to_vnode_offs=_slice(7),
        req_to_vnode_lens=_slice(8),
        mid_o=mid_o,
        mid_lse=mid_lse,
        phase_node_nums=phase_node_nums,
        phase_node_offsets=phase_node_offsets,
        q_tile_sizes=list(Q_TILE_SIZE_PER_PHASE),
        kv_tile_sizes=list(params.TSKs),
    )


# ---------------------------------------------------------------------------
# AttentionContext — writes K/V at slot indices, then runs FastTree's kernel.
# ---------------------------------------------------------------------------


@dataclass
class FastTreeAttentionContext(AttentionContext):
    """Per-decode-step state. Built by ``FastTreeBackend.plan_decode_step``;
    reused across all model layers for the same decode step.
    """
    page_table: PageTable
    write_slots: torch.Tensor   # [B*K] int64 — flat slot to write K/V into
    sm_scale: float
    meta: _FTMeta
    out: torch.Tensor           # scratch [B*K, num_qo_heads, head_dim]

    def attend(self, q, k, v, layer_idx):
        num_kv_heads = self.page_table.head_num
        head_dim = self.page_table.head_dim
        num_heads = q.shape[-1] // head_dim
        kv = self.page_table.kv_cache_at_layer[layer_idx]
        # kv shape: [2, max_pages, page_size, num_kv_heads, head_dim].
        # Both kv[0] and kv[1] are contiguous; view as flat slot-indexed.
        max_pages = kv.shape[1]
        page_size = kv.shape[2]
        flat_len = max_pages * page_size
        K_buf = kv[0].view(flat_len, num_kv_heads, head_dim)
        V_buf = kv[1].view(flat_len, num_kv_heads, head_dim)

        # Append new K/V at the per-beam tail slots.
        k_3d = k.view(-1, num_kv_heads, head_dim)
        v_3d = v.view(-1, num_kv_heads, head_dim)
        K_buf[self.write_slots] = k_3d
        V_buf[self.write_slots] = v_3d

        q_3d = q.view(-1, num_heads, head_dim)
        m = self.meta
        fasttree_decode(
            q_3d,
            K_buf,
            V_buf,
            self.out,
            m.vnode_to_kv_entries,
            m.vnode_to_kv_offs,
            m.vnode_to_kv_lens,
            m.vnode_to_q_entries,
            m.vnode_to_q_offs,
            m.vnode_to_q_lens,
            m.req_to_vnode_entries,
            m.req_to_vnode_offs,
            m.req_to_vnode_lens,
            m.mid_o,
            m.mid_lse,
            m.phase_node_nums,
            m.phase_node_offsets,
            m.q_tile_sizes,
            m.kv_tile_sizes,
            self.sm_scale,
        )
        return self.out.reshape(*q.shape[:-1], num_heads * head_dim)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class _FastTreeWrappers(WrapperBundle):
    """Cached at backend startup. ``out_buf`` is reused across decode steps;
    ``ft_params`` carries autotuned (alpha, beta, gamma) cost-model
    coefficients for ``_tree_heuristic``.

    The ``_heur_*`` slots cache the outputs of ``_tree_heuristic`` /
    ``_compute_parallelism`` between consecutive decode steps. ``_mid_o_buf``
    and ``_mid_lse_buf`` are persistent float32 scratch buffers reused
    across steps instead of being reallocated.
    """
    __slots__ = (
        "ft_params", "out_buf",
        "_heur_key", "_heur_assignments", "_heur_q_tile", "_heur_kv_split",
        "_heur_kv_tsk", "_heur_node_to_reqs",
        "_mid_o_buf", "_mid_lse_buf",
        "_last_dispatch_ms",
    )


@dataclass
class FastTreeBackend:
    name: str = "fasttree"
    use_split_pages: bool = True

    fasttree_params: FastTreeParams | None = None
    KV_SPLIT_SIZES: tuple[int, int] = (1024, 128)
    para_threshs1: tuple[int, int] = (132, 528)
    para_threshs2: tuple[int, int] = (132, 132)

    # Sub-phase trace for plan_decode_step (gated by FT_TRACE_PLAN=1).
    # Each entry is a dict of per-phase ms timings for one step.
    _plan_trace: list = field(default_factory=list, init=False, repr=False)

    def init_wrappers(
        self,
        *,
        workspace_buffer: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        max_cascade_levels: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _FastTreeWrappers:
        wb = _FastTreeWrappers()
        wb.ft_params = self.fasttree_params or FastTreeParams()
        wb.ft_params.set_kv_group_num(num_qo_heads // num_kv_heads)
        # ``out_buf`` is sized once we know B*K at the first decode call; we
        # over-allocate to a generous bound so subsequent steps reuse it.
        wb.out_buf = None
        # Heuristic-loop cache. Populated on the first plan_decode_step
        # then reused while the tree topology + total page count are stable.
        wb._heur_key = None
        wb._heur_assignments = None
        wb._heur_q_tile = None
        wb._heur_kv_split = None
        wb._heur_kv_tsk = None
        wb._heur_node_to_reqs = None
        # Persistent reduction-stage scratch tensors (sized as needed).
        wb._mid_o_buf = None
        wb._mid_lse_buf = None
        # Side channel for plan-trace: _build_metadata writes the
        # dispatch-decision (heuristic loop) elapsed ms here so
        # plan_decode_step can report it separately from index packing.
        wb._last_dispatch_ms = 0.0
        return wb

    def plan_decode_step(
        self,
        *,
        wrappers: _FastTreeWrappers,
        beams_per_prompt: list[list[Beam]],
        current_pos: list[int],
        page_table: PageTable,
        K: int,
        B: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        last_lca_per_prompt: list[int],
    ) -> StepPlan:
        import os as _os
        import time as _time
        ps = page_size
        trace_on = bool(int(_os.environ.get("FT_TRACE_PLAN", "0")))
        if trace_on:
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()

        # ---- Gather per-prompt prefixes + per-beam tails (by reference). ----
        # Aliasing is safe within one plan_decode_step: pages_prefix /
        # pages_tail are not mutated until the next CoW. Avoiding the
        # ``prefix + tail`` concatenation saves ~M element copies.
        shared_prefix_per_prompt: list[list[int]] = []
        tails_per_beam_per_prompt: list[list[list[int]]] = []
        for b in range(B):
            bp_b = beams_per_prompt[b]
            shared_prefix_per_prompt.append(bp_b[0].pages_prefix)
            tails_per_beam_per_prompt.append([beam.pages_tail for beam in bp_b])
        if trace_on:
            _t_gather = _time.perf_counter()

        # ---- Build the combined page-level radix tree. ----
        # All K beams of a prompt share ``pages_prefix`` by construction
        # (split-pages form). The builder aliases the prefix and only
        # walks the divergent tail portion — mirror of bs_kernel's LCA
        # cache, applied to fasttree's radix tree.
        tree_info, node_pages = _build_combined_radix_tree_pages(
            shared_prefix_per_prompt, tails_per_beam_per_prompt, K,
        )
        if trace_on:
            _t_radix = _time.perf_counter()

        # ---- Determine each leaf's partial-last-page count + expand to slots.
        leaf_partial_last: dict[int, int] = {}
        for i, n in enumerate(tree_info):
            if n.num_children == 0 and len(n.requests) == 1 and node_pages[i]:
                rid = n.requests[0]
                b_idx = rid // K
                pos = current_pos[b_idx]
                leaf_partial_last[i] = pos % ps + 1
        if trace_on:
            _t_partial = _time.perf_counter()
        node_slots = _expand_pages_to_slots(
            tree_info, node_pages, ps, leaf_partial_last,
        )
        if trace_on:
            _t_slots = _time.perf_counter()

        # ---- Build FastTree metadata for the kernel. ----
        # Cache key: (n_nodes, B, K, total_pages). Invalidates exactly at
        # page boundaries (when any beam appends a new tail page) and on
        # structural change. The heuristic decision is stable between page
        # boundaries — root seqlen is constant and leaf seqlens grow by ≤1
        # token/step, well below the threshold for assignment flips.
        total_pages = 0
        for b in range(B):
            total_pages += len(shared_prefix_per_prompt[b])
            for tl in tails_per_beam_per_prompt[b]:
                total_pages += len(tl)
        cache_key = (len(tree_info), B, K, total_pages)
        meta = _build_metadata(
            tree_info=tree_info,
            node_slots=node_slots,
            batch_size=B * K,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            KV_SPLIT_SIZES=list(self.KV_SPLIT_SIZES),
            para_threshs1=list(self.para_threshs1),
            para_threshs2=list(self.para_threshs2),
            params=wrappers.ft_params,
            device=device,
            wrappers=wrappers,
            cache_key=cache_key,
        )
        if trace_on:
            torch.cuda.synchronize()
            _t_meta = _time.perf_counter()

        # ---- Build write_slots[B*K]: where to write each beam's new K/V. ----
        write_slots: list[int] = []
        for b in range(B):
            pos = current_pos[b]
            off = pos % ps
            pli = pos // ps
            bp_b = beams_per_prompt[b]
            prefix_len = len(bp_b[0].pages_prefix)
            tail_idx = pli - prefix_len
            for beam in bp_b:
                page = beam.pages_tail[tail_idx]
                write_slots.append(page * ps + off)
        write_slots_t = torch.tensor(
            write_slots, dtype=torch.int64, device=device,
        )
        if trace_on:
            torch.cuda.synchronize()
            _t_write = _time.perf_counter()

        # ---- Allocate / reuse the per-step output buffer. ----
        if wrappers.out_buf is None or wrappers.out_buf.shape[0] < B * K:
            wrappers.out_buf = torch.empty(
                (B * K, num_qo_heads, head_dim),
                dtype=dtype, device=device,
            )
        out_buf = wrappers.out_buf[: B * K]

        sm_scale = 1.0 / (head_dim ** 0.5)
        ctx = FastTreeAttentionContext(
            page_table=page_table,
            write_slots=write_slots_t,
            sm_scale=sm_scale,
            meta=meta,
            out=out_buf,
        )
        if trace_on:
            meta_ms = (_t_meta - _t_slots) * 1000.0
            dispatch_ms = float(getattr(wrappers, "_last_dispatch_ms", 0.0))
            self._plan_trace.append({
                "gather_ms":   (_t_gather  - _t0) * 1000.0,
                "radix_ms":    (_t_radix   - _t_gather) * 1000.0,
                "partial_ms":  (_t_partial - _t_radix) * 1000.0,
                "slots_ms":    (_t_slots   - _t_partial) * 1000.0,
                # meta_ms covers both the dispatch decision (heuristic
                # loop) and the index packing; dispatch_ms is the slice
                # of meta_ms spent in tree_heuristic + compute_parallelism
                # (~0 on cache-hit steps). pack_ms = meta_ms - dispatch_ms.
                "meta_ms":     meta_ms,
                "dispatch_ms": dispatch_ms,
                "pack_ms":     meta_ms - dispatch_ms,
                "write_ms":    (_t_write   - _t_meta) * 1000.0,
                "n_nodes":     len(tree_info),
            })
        # FastTree doesn't reorder beams; identity beam order.
        return StepPlan(ctx=ctx, beam_order_per_prompt=None, pick=None)


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def beam_search(
    model,
    config,
    prompt_ids: list[list[int]],
    max_new_tokens: int,
    beam_width: int,
    *,
    page_size: int = 16,
    max_num_pages: int = 2048,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    return_timings: bool = False,
    return_phase_timings: bool = False,
    fasttree_params: FastTreeParams | None = None,
    KV_SPLIT_SIZES: tuple[int, int] = (1024, 128),
    select_at_prefill: PrefillSelect = standard_prefill_select,
    select_at_decode: DecodeSelect = standard_decode_select,
):
    """Cross-prompt batched FastTree beam search on the unified page-driver.

    All B prompts feed one ``fasttree_decode`` launch per decode step. The
    radix tree is built at PAGE level; the kernel reads K/V via the
    flattened ``kv[0]`` / ``kv[1]`` views of the PageTable's
    ``[2, max_pages, page_size, num_kv_heads, head_dim]`` layout.

    Returns ``list[list[Beam]]`` (outer = per prompt, inner sorted by
    cum_log_prob). With ``return_timings=True`` returns
    ``(beams, timings)``; with ``return_phase_timings=True`` the timings
    also include cow / plan / forward / topk / fork breakdowns.
    """
    backend = FastTreeBackend(
        fasttree_params=fasttree_params,
        KV_SPLIT_SIZES=KV_SPLIT_SIZES,
    )
    result = _shared_beam_search(
        model, config, prompt_ids, max_new_tokens, beam_width,
        backend=backend,
        page_size=page_size,
        max_num_pages=max_num_pages,
        device=device,
        dtype=dtype,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        select_at_prefill=select_at_prefill,
        select_at_decode=select_at_decode,
    )
    import os as _os
    if int(_os.environ.get("FT_TRACE_PLAN", "0")):
        _dump_ft_plan_trace(backend._plan_trace)
    return result


def _dump_ft_plan_trace(trace: list) -> None:
    """Print sub-phase summary of fasttree plan_decode_step."""
    import statistics
    if not trace:
        print("[fasttree plan-trace] no entries")
        return
    n = len(trace)
    fields = (
        "gather_ms", "radix_ms", "partial_ms",
        "slots_ms", "meta_ms", "dispatch_ms", "pack_ms", "write_ms",
    )
    print(f"\n[fasttree plan-trace] {n} steps")
    n_nodes = [r["n_nodes"] for r in trace]
    print(
        f"  tree nodes per step: mean={sum(n_nodes)/n:.1f} "
        f"min={min(n_nodes)} max={max(n_nodes)}"
    )
    print(f"  {'phase':<14} {'mean':>10} {'median':>10} {'p90':>10} {'p99':>10} {'total':>12}")
    for f in fields:
        xs = [r[f] for r in trace]
        xs_sorted = sorted(xs)
        mean = sum(xs) / n
        med = statistics.median(xs)
        p90 = xs_sorted[int(0.9 * (n - 1))]
        p99 = xs_sorted[int(0.99 * (n - 1))]
        total = sum(xs)
        print(
            f"  {f:<14} {mean:>10.3f} {med:>10.3f} "
            f"{p90:>10.3f} {p99:>10.3f} {total:>12.2f}"
        )

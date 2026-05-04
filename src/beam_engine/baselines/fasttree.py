"""Baseline 3 — FastTree (MLSys'25, arXiv:2502.18030).

FastTree replaces the tree-mask approach (baseline 2) with an explicit radix
tree over the KV cache. Its Triton kernel runs a per-tree-node decode in
stage 1 — each "vnode" (a (kv_chunk, q_tile) pair) is one CTA — then merges
per-request partials in stage 2 with online softmax. The kernel-tile
parameters and the K-vs-Q split decision per node are picked by a cost
heuristic (``_tree_heuristic``) at plan time.

For beam search we feed FastTree a radix tree built each decode step from
the beams' KV-slot paths: the longest common slot prefix is one node, and
each diverging subtree continues recursively. Right after prefill this is
just one root (prompt) + K leaves (one per beam); after a fork that keeps
several beams sharing a parent's freshly-allocated slot, intermediate
nodes appear naturally.

The upstream ``fasttree_preparation`` assumes each tree node owns a
contiguous slice ``[KV_ptrs[i], KV_ptrs[i+1])`` of a single flat KV tensor.
Our beam-search slots are interleaved (every step writes K scattered
positions), so we port that preparation here but feed actual slot indices
into ``vnode_to_kv_entries`` directly. Stage 2 is unchanged.
"""

from __future__ import annotations

import queue as _queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

from ..models.attention import AttentionContext


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
# Radix-tree construction over beam KV-slot paths
# ---------------------------------------------------------------------------


def _build_radix_tree(
    beam_paths: list[list[int]],
) -> tuple[list[KVTreeNode], list[list[int]]]:
    """Compress K beam slot-paths into a radix tree.

    Returns (tree_info, node_slots):
      tree_info[i].parent / .seqlen / .num_children / .requests
      node_slots[i] — physical slot indices owned by node i (in KV order).
    All beam paths must share the same length (true after prefill).
    """
    assert beam_paths and all(len(p) == len(beam_paths[0]) for p in beam_paths)
    K = len(beam_paths)
    nodes: list[KVTreeNode] = []
    node_slots: list[list[int]] = []

    def walk(beams: list[int], depth: int, parent_id: int) -> None:
        path_len = len(beam_paths[beams[0]])
        slots: list[int] = []
        d = depth
        # Extend run as long as every beam in this group agrees on the slot.
        while d < path_len:
            slot = beam_paths[beams[0]][d]
            same = True
            for b in beams:
                if beam_paths[b][d] != slot:
                    same = False
                    break
            if not same:
                break
            slots.append(slot)
            d += 1

        n = KVTreeNode()
        n.parent = parent_id
        n.id = len(nodes)
        n.seqlen = len(slots)
        n.num_children = 0
        n.requests = list(beams)  # ancestors store the union of leaf reqs
        my_id = n.id
        nodes.append(n)
        node_slots.append(slots)

        if d == path_len:
            return  # leaf

        groups: dict[int, list[int]] = {}
        for b in beams:
            groups.setdefault(beam_paths[b][d], []).append(b)
        for sub_beams in groups.values():
            walk(sub_beams, d, my_id)
            nodes[my_id].num_children += 1

    walk(list(range(K)), 0, -1)
    return nodes, node_slots


# ---------------------------------------------------------------------------
# FastTree-style metadata build (port of ``fasttree_preparation`` that takes
# scattered slot indices instead of contiguous KV_ptrs ranges).
# ---------------------------------------------------------------------------


@dataclass
class _FTMeta:
    """All tensors + scalars that fasttree_decode needs for a step."""
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
) -> _FTMeta:
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

    Q_TILE_SIZE = Q_TILE_SIZE_PER_PHASE[0]
    vnode_to_kv_entries: list[int] = []
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
        curr_KV_indices = list(node_slots[node])
        kv_len = tree_info[node].seqlen
        while node_assignments[node] == 1:
            node = tree_info[node].parent
            kv_len += tree_info[node].seqlen
            curr_KV_indices = list(node_slots[node]) + curr_KV_indices

        acc_kv_len = len(vnode_to_kv_entries)
        vnode_to_kv_entries.extend(curr_KV_indices)

        kv_vnode_count = (kv_len - 1) // KV_SPLIT_SIZE + 1
        q_vnode_count = (req_num - 1) // Q_TILE_SIZE + 1
        for kv_vnode_id in range(kv_vnode_count):
            acc_q_len = len(vnode_to_q_entries)
            for req in node_to_reqs[i]:
                vnode_to_q_entries.append(req)
            split_kv_off = kv_vnode_id * KV_SPLIT_SIZE
            vnode_kv_len = min(split_kv_off + KV_SPLIT_SIZE, kv_len) - split_kv_off
            for q_vnode_id in range(q_vnode_count):
                split_q_off = q_vnode_id * Q_TILE_SIZE
                vnode_q_len = min(split_q_off + Q_TILE_SIZE, req_num) - split_q_off
                vnode_to_kv_offs.append(acc_kv_len + split_kv_off)
                vnode_to_kv_lens.append(vnode_kv_len)
                vnode_to_q_offs.append(acc_q_len + split_q_off)
                vnode_to_q_lens.append(vnode_q_len)

    # req → vnode reduction metadata (stage 2)
    per_req: list[list[int]] = [[] for _ in range(batch_size)]
    for vidx, q in enumerate(vnode_to_q_entries):
        per_req[q].append(vidx)
    req_to_vnode_offs: list[int] = []
    req_to_vnode_lens: list[int] = []
    offset = 0
    for r in range(batch_size):
        req_to_vnode_offs.append(offset)
        req_to_vnode_lens.append(len(per_req[r]))
        offset += len(per_req[r])
    req_to_vnode_entries = [v for sub in per_req for v in sub]

    # Phase split + reorder vnodes by Q-tile size (matches fasttree.py).
    threshold = Q_TILE_SIZE_PER_PHASE[1]
    above = [i for i, val in enumerate(vnode_to_q_lens) if val > threshold]
    below = [i for i, val in enumerate(vnode_to_q_lens) if val <= threshold]
    new_order = above + below
    phase_node_nums = [len(above), len(below)]
    phase_node_offsets = [0, len(above)]
    vnode_to_q_lens = [vnode_to_q_lens[i] for i in new_order]
    vnode_to_q_offs = [vnode_to_q_offs[i] for i in new_order]
    vnode_to_kv_lens = [vnode_to_kv_lens[i] for i in new_order]
    vnode_to_kv_offs = [vnode_to_kv_offs[i] for i in new_order]

    # Padding (kernel reads up to KV_TILE_SIZE entries past end with masking).
    vnode_to_kv_entries = vnode_to_kv_entries + [-1] * 64
    req_to_vnode_entries = req_to_vnode_entries + [-1] * 64

    def _t(arr):
        return torch.tensor(arr, dtype=torch.int32, device=device)

    n_vnodes = max(1, sum(phase_node_nums))
    n_q_entries = max(1, len(vnode_to_q_entries))
    mid_o = torch.empty(
        (n_q_entries, num_qo_heads, head_dim), dtype=torch.float32, device=device,
    )
    mid_lse = torch.empty(
        (n_q_entries, num_qo_heads), dtype=torch.float32, device=device,
    )

    return _FTMeta(
        vnode_to_kv_entries=_t(vnode_to_kv_entries),
        vnode_to_kv_offs=_t(vnode_to_kv_offs),
        vnode_to_kv_lens=_t(vnode_to_kv_lens),
        vnode_to_q_entries=_t(vnode_to_q_entries),
        vnode_to_q_offs=_t(vnode_to_q_offs),
        vnode_to_q_lens=_t(vnode_to_q_lens),
        req_to_vnode_entries=_t(req_to_vnode_entries),
        req_to_vnode_offs=_t(req_to_vnode_offs),
        req_to_vnode_lens=_t(req_to_vnode_lens),
        mid_o=mid_o,
        mid_lse=mid_lse,
        phase_node_nums=phase_node_nums,
        phase_node_offsets=phase_node_offsets,
        q_tile_sizes=list(Q_TILE_SIZE_PER_PHASE),
        kv_tile_sizes=list(params.TSKs),
    )


# ---------------------------------------------------------------------------
# Attention contexts — separate prefill (FlashInfer ragged) and decode (FastTree)
# ---------------------------------------------------------------------------


@dataclass
class _PrefillCtx(AttentionContext):
    """Standard ragged-prefill: writes prompt KV into the fused buffer at
    ``write_indices`` and runs FlashInfer's causal ragged kernel.
    """
    kv_cache: list[torch.Tensor]      # per layer: [max_tokens, 2, num_kv_heads, head_dim]
    write_indices: torch.Tensor       # [L_p] int64
    num_kv_heads: int
    head_dim: int
    num_qo_heads: int
    wrapper: BatchPrefillWithRaggedKVCacheWrapper

    def attend(self, q, k, v, layer_idx):
        kv = self.kv_cache[layer_idx]
        k_3d = k.view(-1, self.num_kv_heads, self.head_dim)
        v_3d = v.view(-1, self.num_kv_heads, self.head_dim)
        kv[self.write_indices, 0] = k_3d
        kv[self.write_indices, 1] = v_3d
        q_3d = q.view(-1, self.num_qo_heads, self.head_dim)
        out = self.wrapper.run(q_3d, kv[:, 0], kv[:, 1])
        return out.reshape(*q.shape[:-1], self.num_qo_heads * self.head_dim)


@dataclass
class FastTreeAttentionContext(AttentionContext):
    """Per-decode-step FastTree state. Built once; reused across all layers."""
    kv_cache: list[torch.Tensor]      # per layer: [max_tokens, 2, num_kv_heads, head_dim]
    write_indices: torch.Tensor       # [K] int64 — slot to write per beam this step
    num_kv_heads: int
    head_dim: int
    num_qo_heads: int
    sm_scale: float
    meta: _FTMeta
    out: torch.Tensor                  # scratch [K, num_qo_heads, head_dim]

    def attend(self, q, k, v, layer_idx):
        kv = self.kv_cache[layer_idx]
        k_3d = k.view(-1, self.num_kv_heads, self.head_dim)
        v_3d = v.view(-1, self.num_kv_heads, self.head_dim)
        kv[self.write_indices, 0] = k_3d
        kv[self.write_indices, 1] = v_3d

        q_3d = q.view(-1, self.num_qo_heads, self.head_dim)
        # FastTree wants K and V as [total_kv_tokens, num_kv_heads, head_dim] —
        # use the appropriate slice of the fused buffer (shared across layers
        # in shape, sliced per call by layer).
        K_buf = kv[:, 0].contiguous() if not kv[:, 0].is_contiguous() else kv[:, 0]
        V_buf = kv[:, 1].contiguous() if not kv[:, 1].is_contiguous() else kv[:, 1]

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
        return self.out.reshape(*q.shape[:-1], self.num_qo_heads * self.head_dim)


# ---------------------------------------------------------------------------
# Beam state + per-prompt driver
# ---------------------------------------------------------------------------


@dataclass
class Beam:
    token_ids: list[int]
    cum_log_prob: float
    path: list[int]   # ordered fused-buffer indices on this beam's KV path


def _beam_search_single(
    model,
    config,
    prompt_ids: list[int],
    max_new_tokens: int,
    beam_width: int,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    timings: dict | None = None,
    fasttree_params: FastTreeParams | None = None,
    KV_SPLIT_SIZES: tuple[int, int] = (1024, 128),
    para_threshs1: tuple[int, int] = (132, 528),
    para_threshs2: tuple[int, int] = (132, 132),
) -> list[Beam]:
    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers

    K = beam_width
    L_p = len(prompt_ids)
    max_tokens = L_p + K * max_new_tokens + K

    kv_cache = [
        torch.zeros(
            (max_tokens, 2, num_kv_heads, head_dim), dtype=dtype, device=device,
        )
        for _ in range(num_layers)
    ]

    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device,
    )
    prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD",
    )

    # ----- Prefill -----
    prefill_write = torch.arange(L_p, dtype=torch.long, device=device)
    qo_indptr = torch.tensor([0, L_p], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, L_p], dtype=torch.int32, device=device)
    prefill_wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        causal=True,
    )
    pre_ctx = _PrefillCtx(
        kv_cache=kv_cache,
        write_indices=prefill_write,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_qo_heads=num_qo_heads,
        wrapper=prefill_wrapper,
    )
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    positions = torch.arange(L_p, device=device).unsqueeze(0)

    if timings is not None:
        torch.cuda.synchronize()
        t_pre = time.perf_counter()

    with torch.no_grad():
        hidden = model.forward(input_ids=input_ids, positions=positions, ctx=pre_ctx)
        logits = model.compute_logits(hidden[:, -1, :])
        log_probs = F.log_softmax(logits, dim=-1)
        topk_log_probs, topk_ids = log_probs.topk(K, dim=-1)
        topk_log_probs = topk_log_probs.squeeze(0)
        topk_ids = topk_ids.squeeze(0)

    if timings is not None:
        torch.cuda.synchronize()
        timings["prefill_ms"] = (time.perf_counter() - t_pre) * 1000.0

    prefix_path = list(range(L_p))
    beams = [
        Beam(
            token_ids=[topk_ids[i].item()],
            cum_log_prob=topk_log_probs[i].item(),
            path=list(prefix_path),
        )
        for i in range(K)
    ]
    kv_total_len = L_p
    current_pos = L_p

    sm_scale = 1.0 / (head_dim ** 0.5)
    out_buf = torch.empty(
        (K, num_qo_heads, head_dim), dtype=dtype, device=device,
    )

    # ----- Decode loop -----
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if timings is not None:
                torch.cuda.synchronize()
                t_step = time.perf_counter()

            # 1) Allocate K new slots, one per beam.
            new_slots = list(range(kv_total_len, kv_total_len + K))
            kv_total_len += K
            for k in range(K):
                beams[k].path.append(new_slots[k])

            # 2) Build radix tree from the beams' slot paths.
            paths = [b.path for b in beams]
            tree_info, node_slots = _build_radix_tree(paths)

            # 3) Build FastTree metadata + heuristic-tune the tile sizes.
            params = fasttree_params or FastTreeParams()
            params.set_kv_group_num(num_qo_heads // num_kv_heads)
            meta = _build_metadata(
                tree_info,
                node_slots,
                batch_size=K,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                KV_SPLIT_SIZES=list(KV_SPLIT_SIZES),
                para_threshs1=list(para_threshs1),
                para_threshs2=list(para_threshs2),
                params=params,
                device=torch.device(device),
            )

            ctx = FastTreeAttentionContext(
                kv_cache=kv_cache,
                write_indices=torch.tensor(
                    new_slots, dtype=torch.long, device=device,
                ),
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_qo_heads=num_qo_heads,
                sm_scale=sm_scale,
                meta=meta,
                out=out_buf,
            )

            beam_input = torch.tensor(
                [[b.token_ids[-1]] for b in beams], dtype=torch.long, device=device,
            )
            beam_positions = torch.full((K, 1), current_pos, device=device)

            hidden = model.forward(
                input_ids=beam_input, positions=beam_positions, ctx=ctx,
            )
            logits = model.compute_logits(hidden[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)

            # 4) Top-K + fork.
            vocab_size = logits.shape[-1]
            cum_probs = torch.tensor(
                [b.cum_log_prob for b in beams], device=device, dtype=torch.float32,
            )
            scores = cum_probs[:, None] + log_probs.float()
            topk_scores, topk_flat = scores.reshape(-1).topk(K)
            ids = torch.stack(
                (topk_flat // vocab_size, topk_flat % vocab_size), dim=0,
            ).tolist()
            parent_ids = ids[0]
            new_token_ids = ids[1]
            scores_list = topk_scores.tolist()

            new_beams: list[Beam] = []
            for i in range(K):
                pid = parent_ids[i]
                new_beams.append(
                    Beam(
                        token_ids=beams[pid].token_ids + [new_token_ids[i]],
                        cum_log_prob=scores_list[i],
                        path=list(beams[pid].path),
                    )
                )
            beams = new_beams
            current_pos += 1

            if timings is not None:
                torch.cuda.synchronize()
                timings["decode_step_ms"].append(
                    (time.perf_counter() - t_step) * 1000.0
                )

    beams.sort(key=lambda b: b.cum_log_prob, reverse=True)
    return beams


def beam_search(
    model,
    config,
    prompt_ids: list[list[int]],
    max_new_tokens: int,
    beam_width: int,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    return_timings: bool = False,
    fasttree_params: FastTreeParams | None = None,
    KV_SPLIT_SIZES: tuple[int, int] = (1024, 128),
    para_threshs1: tuple[int, int] = (132, 528),
    para_threshs2: tuple[int, int] = (132, 132),
):
    """FastTree beam search. Multi-prompt batches run sequentially (one fused
    KV buffer per prompt), matching the tree-attention baseline's contract.
    """
    if return_timings:
        agg = {"prefill_ms": 0.0, "decode_step_ms": []}
        all_beams: list[list[Beam]] = []
        for p in prompt_ids:
            t = {"prefill_ms": 0.0, "decode_step_ms": []}
            beams = _beam_search_single(
                model, config, p, max_new_tokens, beam_width,
                device=device, dtype=dtype, timings=t,
                fasttree_params=fasttree_params,
                KV_SPLIT_SIZES=KV_SPLIT_SIZES,
                para_threshs1=para_threshs1,
                para_threshs2=para_threshs2,
            )
            all_beams.append(beams)
            agg["prefill_ms"] += t["prefill_ms"]
            agg["decode_step_ms"].extend(t["decode_step_ms"])
        return all_beams, agg
    return [
        _beam_search_single(
            model, config, p, max_new_tokens, beam_width,
            device=device, dtype=dtype,
            fasttree_params=fasttree_params,
            KV_SPLIT_SIZES=KV_SPLIT_SIZES,
            para_threshs1=para_threshs1,
            para_threshs2=para_threshs2,
        )
        for p in prompt_ids
    ]

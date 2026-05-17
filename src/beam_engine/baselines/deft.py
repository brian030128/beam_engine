"""Baseline — DeFT (MLSys'25, arXiv:2404.00242), integrated into our
shared page_driver beam-search loop.

DeFT's contribution is the "split-by-node" two-stage Triton attention
kernel: one CTA per (head, node) processes up to ``BLOCK_M=32`` queries
that read this node's KV, emitting a partial output + LSE per
(node, query); stage 2 merges per-query partials via online softmax.

This backend builds a *page-level* radix tree across the B*K beams
(reusing ``fasttree._build_combined_radix_tree_pages`` /
``_expand_pages_to_slots``) and converts it into DeFT's flat-array
metadata format:

  KV_indices         : concatenated per-node slot arrays
  KV_indices_offset  : per-node start into KV_indices
  KV_len             : per-node slot count
  KVMapQ_List        : per-node lists of query (= beam) indices
  KVMapQ_List_Offset : per-node start into KVMapQ_List
  KVMapQ_List_Len    : per-node query count

DeFT's stage-1 kernel only processes ``BLOCK_M=32`` queries per (head,
node) CTA — nodes with more readers (e.g. the per-prompt sub-root at
K=64) would silently drop the overflow. We split such nodes into
``ceil(n_readers / BLOCK_M)`` *virtual* entries that share the same KV
range (same KV_indices_offset / KV_len) but partition the readers
across non-overlapping KVMapQ_List slices. Stage 2 merges as usual.

The vendored kernel lives in ``baselines/_deft_kernel/`` (copied
verbatim from DeFT's artifact, since DeFT's package pins
torch==2.5.1 / python>=3.12 and can't be co-installed with our env).
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
from ._deft_kernel import tree_attention_fwd as _deft_tree_attn_fwd
from .fasttree import (
    _build_combined_radix_tree_pages,
    _expand_pages_to_slots,
)


# DeFT's stage-1 kernel processes BLOCK_M queries per (head, node) CTA.
# Must match ``BLOCK_M`` in
# ``_deft_kernel/tree_attention.py:DeFT_splitBynode_Triton_stage1``.
_DEFT_BLOCK_M = 32


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class _DeftMeta:
    KV_indices: torch.Tensor          # (sum_node_slots,) int32
    KV_indices_offset: torch.Tensor   # (kv_num,) int32
    KV_len: torch.Tensor              # (kv_num,) int32
    KVMapQ_List: torch.Tensor         # (sum_node_queries,) int32
    KVMapQ_List_Offset: torch.Tensor  # (kv_num,) int32
    KVMapQ_List_Len: torch.Tensor     # (kv_num,) int32


def _build_deft_metadata(
    tree_info,
    node_slots: list[np.ndarray],
    device: torch.device,
) -> _DeftMeta:
    """Pack the page-radix tree into DeFT's flat-array metadata.

    Skips nodes with zero seqlen or zero readers (virtual root). Splits
    nodes whose reader count exceeds ``_DEFT_BLOCK_M`` into multiple
    virtual entries sharing the same KV range; this is invisible to the
    kernel — stage 2 merges per-query partials regardless of how many
    (node, query-chunk) pairs contributed.
    """
    kv_indices_chunks: list[np.ndarray] = []
    kv_indices_offset: list[int] = []
    kv_len: list[int] = []
    kvmapq_chunks: list[np.ndarray] = []
    kvmapq_offset: list[int] = []
    kvmapq_len: list[int] = []

    kv_acc = 0      # running total slots emitted
    mapq_acc = 0    # running total query-entries emitted

    for i, n in enumerate(tree_info):
        n_slots = node_slots[i].size
        reqs = n.requests
        if n_slots == 0 or not reqs:
            continue
        kv_start = kv_acc
        kv_indices_chunks.append(node_slots[i])
        kv_acc += n_slots
        reqs_arr = np.asarray(reqs, dtype=np.int32)
        n_reqs = reqs_arr.size
        # Split readers into BLOCK_M-sized chunks; emit one virtual
        # kv_num entry per chunk. All chunks share (kv_start, n_slots).
        for off in range(0, n_reqs, _DEFT_BLOCK_M):
            chunk = reqs_arr[off : off + _DEFT_BLOCK_M]
            kvmapq_chunks.append(chunk)
            kvmapq_offset.append(mapq_acc)
            kvmapq_len.append(int(chunk.size))
            mapq_acc += chunk.size
            kv_indices_offset.append(kv_start)
            kv_len.append(n_slots)

    if kv_indices_chunks:
        kv_indices_np = np.concatenate(kv_indices_chunks)
    else:
        kv_indices_np = np.empty(0, dtype=np.int32)
    if kvmapq_chunks:
        kvmapq_np = np.concatenate(kvmapq_chunks)
    else:
        kvmapq_np = np.empty(0, dtype=np.int32)

    kv_indices_offset_np = np.asarray(kv_indices_offset, dtype=np.int32)
    kv_len_np = np.asarray(kv_len, dtype=np.int32)
    kvmapq_offset_np = np.asarray(kvmapq_offset, dtype=np.int32)
    kvmapq_len_np = np.asarray(kvmapq_len, dtype=np.int32)

    # Single H2D for all int32 metadata (same pattern as fasttree.py).
    arrays = (
        kv_indices_np,           # 0
        kv_indices_offset_np,    # 1
        kv_len_np,               # 2
        kvmapq_np,               # 3
        kvmapq_offset_np,        # 4
        kvmapq_len_np,           # 5
    )
    sizes = [a.size for a in arrays]
    starts = [0]
    for s in sizes:
        starts.append(starts[-1] + s)
    big = (
        np.concatenate(arrays)
        if starts[-1] > 0
        else np.empty(0, dtype=np.int32)
    )
    big_t = torch.from_numpy(big).to(device, non_blocking=True)

    def _slice(i):
        return big_t[starts[i] : starts[i + 1]]

    return _DeftMeta(
        KV_indices=_slice(0),
        KV_indices_offset=_slice(1),
        KV_len=_slice(2),
        KVMapQ_List=_slice(3),
        KVMapQ_List_Offset=_slice(4),
        KVMapQ_List_Len=_slice(5),
    )


# ---------------------------------------------------------------------------
# AttentionContext
# ---------------------------------------------------------------------------


@dataclass
class DeftAttentionContext(AttentionContext):
    """Per-decode-step state. Built by ``DeftBackend.plan_decode_step``;
    reused across all model layers for the same decode step.
    """
    page_table: PageTable
    write_slots: torch.Tensor   # [B*K] int64 — where to write new K/V
    meta: _DeftMeta
    out: torch.Tensor           # scratch [B*K, num_qo_heads, head_dim]

    def attend(self, q, k, v, layer_idx):
        num_kv_heads = self.page_table.head_num
        head_dim = self.page_table.head_dim
        num_heads = q.shape[-1] // head_dim
        kv = self.page_table.kv_cache_at_layer[layer_idx]
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
        _deft_tree_attn_fwd(
            q_3d,
            K_buf,
            V_buf,
            self.out,
            m.KV_indices,
            m.KV_indices_offset,
            m.KV_len,
            m.KVMapQ_List,
            m.KVMapQ_List_Offset,
            m.KVMapQ_List_Len,
        )
        return self.out.reshape(*q.shape[:-1], num_heads * head_dim)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class _DeftWrappers(WrapperBundle):
    __slots__ = ("out_buf",)


@dataclass
class DeftBackend:
    name: str = "deft"
    use_split_pages: bool = True

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
    ) -> _DeftWrappers:
        wb = _DeftWrappers()
        wb.out_buf = None
        return wb

    def plan_decode_step(
        self,
        *,
        wrappers: _DeftWrappers,
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
        ps = page_size

        # ---- Gather per-prompt prefixes + per-beam tails (by reference).
        shared_prefix_per_prompt: list[list[int]] = []
        tails_per_beam_per_prompt: list[list[list[int]]] = []
        for b in range(B):
            bp_b = beams_per_prompt[b]
            shared_prefix_per_prompt.append(bp_b[0].pages_prefix)
            tails_per_beam_per_prompt.append([beam.pages_tail for beam in bp_b])

        # ---- Build the combined page-level radix tree (shared w/ fasttree).
        tree_info, node_pages = _build_combined_radix_tree_pages(
            shared_prefix_per_prompt, tails_per_beam_per_prompt, K,
        )

        # ---- Leaf partial-last-page handling, then expand pages→slots.
        leaf_partial_last: dict[int, int] = {}
        for i, n in enumerate(tree_info):
            if n.num_children == 0 and len(n.requests) == 1 and node_pages[i]:
                rid = n.requests[0]
                b_idx = rid // K
                pos = current_pos[b_idx]
                leaf_partial_last[i] = pos % ps + 1
        node_slots = _expand_pages_to_slots(
            tree_info, node_pages, ps, leaf_partial_last,
        )

        # ---- Pack into DeFT's flat-array metadata.
        meta = _build_deft_metadata(tree_info, node_slots, device)

        # ---- Build write_slots[B*K]: where to append the new K/V.
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

        # ---- Allocate / reuse the per-step output buffer.
        if wrappers.out_buf is None or wrappers.out_buf.shape[0] < B * K:
            wrappers.out_buf = torch.empty(
                (B * K, num_qo_heads, head_dim),
                dtype=dtype, device=device,
            )
        out_buf = wrappers.out_buf[: B * K]

        ctx = DeftAttentionContext(
            page_table=page_table,
            write_slots=write_slots_t,
            meta=meta,
            out=out_buf,
        )
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
    select_at_prefill: PrefillSelect = standard_prefill_select,
    select_at_decode: DecodeSelect = standard_decode_select,
):
    """Cross-prompt batched DeFT beam search on the unified page-driver.

    All B prompts feed one ``tree_attention_fwd`` (two-stage Triton)
    launch per decode step. The radix tree is built at PAGE level; the
    kernel reads K/V via the flattened ``kv[0]`` / ``kv[1]`` views of
    the PageTable's ``[2, max_pages, page_size, num_kv_heads, head_dim]``
    layout.

    Returns ``list[list[Beam]]`` (outer = per prompt, inner sorted by
    cum_log_prob). With ``return_timings=True`` returns
    ``(beams, timings)``.
    """
    backend = DeftBackend()
    return _shared_beam_search(
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


# ---------------------------------------------------------------------------
# Backend factory for the SGLang e2e bench (tree_batch_decode protocol)
# ---------------------------------------------------------------------------


class DeftBackendForTreeDecode(DeftBackend):
    """Alias surfaced under a more descriptive name so bench_sglang_e2e.py
    can pick it up via factory dict like the other baselines."""
    pass

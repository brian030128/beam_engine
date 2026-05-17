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

**Plan-state cache (LCA cache).** Within a stable-topology period the
per-node page lists and query mapping are all stable — only
single-request leaves' ``KV_len`` grows by 1 per step as the partial
last page fills. We cache the int32 numpy arrays + per-leaf patch
info; cache hits skip the page→slot expansion and array packing,
doing only a vectorised leaf-KV_len patch + one H2D copy. (The radix
tree itself is always rebuilt — for our beam-search workloads the
fast-path radix at ``fasttree._build_radix_tree_pages_single`` is a
few microseconds.)

Cache key follows the fasttree pattern in ``baselines/fasttree.py:801``:
``(len(tree_info), B, K, total_pages)``. This is invariant under the
list-identity churn from page_driver's fork phase (which copies
``parent.pages_tail`` to a fresh list every step for non-assignee
children at ``page_driver.py:726``) — an id-based key gets 0% hit
rate. ``len(tree_info)`` catches CoW (always splits a previously-
shared node into a deeper branch, adding at least one tree node);
``total_pages`` catches page-boundary appends. Disable for A/B with
``BE_DEFT_PLAN_CACHE=0``.
"""

from __future__ import annotations

import os
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
# Metadata + plan cache
# ---------------------------------------------------------------------------


@dataclass
class _DeftMeta:
    KV_indices: torch.Tensor          # (sum_node_slots,) int32
    KV_indices_offset: torch.Tensor   # (kv_num,) int32
    KV_len: torch.Tensor              # (kv_num,) int32
    KVMapQ_List: torch.Tensor         # (sum_node_queries,) int32
    KVMapQ_List_Offset: torch.Tensor  # (kv_num,) int32
    KVMapQ_List_Len: torch.Tensor     # (kv_num,) int32


@dataclass
class _DeftPlanCache:
    """Cacheable arrays for one ``plan_decode_step``.

    All fields are stable within a page-boundary period; the only
    per-step change is ``KV_len_np`` at leaf-entry indices, patched by
    ``_patch_and_h2d``.

    Leaf-entry patch info (parallel int32 arrays of length
    ``n_single_request_leaf_entries``):
      * ``leaf_entry_idxs``  — index into ``KV_len_np``
      * ``leaf_entry_b``     — owning prompt index (for current_pos[b])
      * ``leaf_entry_base``  — ``(n_pages - 1) * page_size`` for the leaf
                                (prior full pages; per-step KV_len adds
                                ``current_pos[b] % page_size + 1``)
    Multi-request leaves and non-leaf entries keep the cached KV_len
    (full ``n_pages * page_size``) unchanged across steps — same as the
    pre-cache behaviour, since multi-request leaves don't get partial
    trim either (all beams sharing the leaf necessarily share off too).
    """
    KV_indices_np: np.ndarray
    KV_indices_offset_np: np.ndarray
    KV_len_np: np.ndarray              # mutable: leaf entries patched per step
    KVMapQ_List_np: np.ndarray
    KVMapQ_List_Offset_np: np.ndarray
    KVMapQ_List_Len_np: np.ndarray
    leaf_entry_idxs: np.ndarray
    leaf_entry_b: np.ndarray
    leaf_entry_base: np.ndarray


def _build_deft_cache_key(
    tree_info: list,
    total_pages: int,
    B: int,
    K: int,
) -> tuple:
    """Fasttree-style key: ``(len(tree_info), B, K, total_pages)``.

    Invariant under page_driver's fork-list churn (CoW always splits a
    shared radix node → ``len(tree_info)`` grows; page-boundary append
    grows ``total_pages``). An id-based key gets 0% hit rate because
    ``page_driver.py:726`` copies ``parent.pages_tail`` to a fresh list
    every step for non-assignee children.
    """
    return (len(tree_info), B, K, total_pages)


def _pack_deft_arrays(
    tree_info,
    node_pages: list[list[int]],
    node_slots: list[np.ndarray],
    K: int,
    page_size: int,
) -> _DeftPlanCache:
    """Pack the radix tree into DeFT's flat-array metadata.

    Splits nodes whose reader count exceeds ``_DEFT_BLOCK_M`` into
    multiple virtual entries sharing the same KV range; stage 2 merges
    per-query partials regardless of how many (node, query-chunk) pairs
    contributed.

    Single-request leaves contribute their *full last page* of slots (no
    partial trim); the cache layer patches their per-step KV_len via
    ``leaf_entry_*`` arrays. The kernel masks beyond KV_len so the extra
    not-yet-written slots are never read.
    """
    kv_indices_chunks: list[np.ndarray] = []
    kv_indices_offset: list[int] = []
    kv_len: list[int] = []
    kvmapq_chunks: list[np.ndarray] = []
    kvmapq_offset: list[int] = []
    kvmapq_len: list[int] = []
    leaf_entry_idxs: list[int] = []
    leaf_entry_b: list[int] = []
    leaf_entry_base: list[int] = []

    kv_acc = 0
    mapq_acc = 0
    entry_idx = 0  # running index into kv_len / kv_indices_offset

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

        # Single-request leaf? If so all its chunks (one chunk for a
        # single reader, since 1 ≤ BLOCK_M) will be patched per step.
        is_single_leaf = (
            n.num_children == 0 and n_reqs == 1 and len(node_pages[i]) > 0
        )
        if is_single_leaf:
            rid = int(reqs_arr[0])
            b_owner = rid // K
            base_len = (len(node_pages[i]) - 1) * page_size

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
            if is_single_leaf:
                # Single-request leaf → exactly one chunk (n_reqs == 1
                # ≤ BLOCK_M). Record patch info for this entry.
                leaf_entry_idxs.append(entry_idx)
                leaf_entry_b.append(b_owner)
                leaf_entry_base.append(base_len)
            entry_idx += 1

    if kv_indices_chunks:
        kv_indices_np = np.concatenate(kv_indices_chunks)
    else:
        kv_indices_np = np.empty(0, dtype=np.int32)
    if kvmapq_chunks:
        kvmapq_np = np.concatenate(kvmapq_chunks)
    else:
        kvmapq_np = np.empty(0, dtype=np.int32)

    return _DeftPlanCache(
        KV_indices_np=kv_indices_np,
        KV_indices_offset_np=np.asarray(kv_indices_offset, dtype=np.int32),
        KV_len_np=np.asarray(kv_len, dtype=np.int32),
        KVMapQ_List_np=kvmapq_np,
        KVMapQ_List_Offset_np=np.asarray(kvmapq_offset, dtype=np.int32),
        KVMapQ_List_Len_np=np.asarray(kvmapq_len, dtype=np.int32),
        leaf_entry_idxs=np.asarray(leaf_entry_idxs, dtype=np.int32),
        leaf_entry_b=np.asarray(leaf_entry_b, dtype=np.int32),
        leaf_entry_base=np.asarray(leaf_entry_base, dtype=np.int32),
    )


def _patch_and_h2d(
    cache: _DeftPlanCache,
    current_pos: list[int],
    page_size: int,
    device: torch.device,
) -> _DeftMeta:
    """Patch leaf KV_lens to the current partial-last-page count, then
    concatenate all int32 arrays and copy to device in a single H2D.

    Mutates ``cache.KV_len_np`` in place — safe because every step
    re-patches all leaf entries (idempotent overwrite).
    """
    if cache.leaf_entry_idxs.size > 0:
        cur_pos_np = np.asarray(current_pos, dtype=np.int64)
        pos_per_leaf = cur_pos_np[cache.leaf_entry_b]
        partial = (pos_per_leaf % page_size + 1).astype(np.int32)
        cache.KV_len_np[cache.leaf_entry_idxs] = (
            cache.leaf_entry_base + partial
        )

    arrays = (
        cache.KV_indices_np,           # 0
        cache.KV_indices_offset_np,    # 1
        cache.KV_len_np,               # 2
        cache.KVMapQ_List_np,          # 3
        cache.KVMapQ_List_Offset_np,   # 4
        cache.KVMapQ_List_Len_np,      # 5
    )
    sizes = [a.size for a in arrays]
    starts = [0]
    for s in sizes:
        starts.append(starts[-1] + s)
    if starts[-1] > 0:
        big = np.concatenate(arrays)
    else:
        big = np.empty(0, dtype=np.int32)
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
    __slots__ = ("out_buf", "plan_cache", "plan_cache_key")


@dataclass
class DeftBackend:
    name: str = "deft"
    use_split_pages: bool = True

    _plan_trace: list = field(default_factory=list, init=False, repr=False)
    # Cache hit/miss counters (visible via the backend handle for the
    # bench harnesses that want to print them, mirroring mlca's pattern).
    _plan_hits: int = field(default=0, init=False, repr=False)
    _plan_misses: int = field(default=0, init=False, repr=False)

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
        wb.plan_cache = None
        wb.plan_cache_key = None
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
        cache_enabled = os.environ.get("BE_DEFT_PLAN_CACHE", "1") != "0"

        # ---- Always build the page-level radix tree. Fast-path at
        # fasttree._build_radix_tree_pages_single is a few microseconds
        # for the common K-distinct-divergence case; the heavy work is
        # in _pack_deft_arrays which the cache skips on hit.
        shared_prefix_per_prompt: list[list[int]] = []
        tails_per_beam_per_prompt: list[list[list[int]]] = []
        total_pages = 0
        for b in range(B):
            bp_b = beams_per_prompt[b]
            prefix = bp_b[0].pages_prefix
            shared_prefix_per_prompt.append(prefix)
            tails = [beam.pages_tail for beam in bp_b]
            tails_per_beam_per_prompt.append(tails)
            total_pages += len(prefix)
            for tl in tails:
                total_pages += len(tl)
        tree_info, node_pages = _build_combined_radix_tree_pages(
            shared_prefix_per_prompt, tails_per_beam_per_prompt, K,
        )

        # ---- Cache lookup, fasttree-style key. Invariant under fork-
        # induced list-identity churn; CoW always splits a previously-
        # shared radix node (len(tree_info) grows) so any content change
        # is caught by the key.
        cache_key = (
            _build_deft_cache_key(tree_info, total_pages, B, K)
            if cache_enabled else None
        )
        cache: _DeftPlanCache | None = None
        if (
            cache_enabled
            and wrappers.plan_cache is not None
            and wrappers.plan_cache_key == cache_key
        ):
            cache = wrappers.plan_cache
            self._plan_hits += 1
        else:
            self._plan_misses += 1
            # Empty leaf_partial_last → _expand_pages_to_slots returns
            # full-page slots for every leaf. _patch_and_h2d sets the
            # per-step KV_len to the correct partial count; the kernel
            # masks beyond KV_len so the extra slots are never read.
            node_slots = _expand_pages_to_slots(
                tree_info, node_pages, ps, {},
            )
            cache = _pack_deft_arrays(tree_info, node_pages, node_slots, K, ps)
            if cache_enabled:
                wrappers.plan_cache = cache
                wrappers.plan_cache_key = cache_key

        meta = _patch_and_h2d(cache, current_pos, ps, device)

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

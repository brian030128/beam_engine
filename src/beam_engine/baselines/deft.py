"""Baseline — DeFT (MLSys'25, arXiv:2404.00242), integrated into our
shared page_driver beam-search loop.

DeFT's contribution is the "split-by-node" two-stage Triton attention
kernel: one CTA per (head, node) processes up to ``BLOCK_M=32`` queries
that read this node's KV, emitting a partial output + LSE per
(node, query); stage 2 merges per-query partials via online softmax.

This backend builds a *page-level* radix tree across the B*K beams
(reusing ``fasttree._build_radix_tree_pages_single`` /
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

**Plan-state cache (per-prompt LCA cache).** The metadata for each
prompt's sub-tree is cached independently. Per-step, each prompt's
key is checked against the cache: hit → reuse the cached arrays;
miss → rebuild just that prompt's sub-tree. The concatenation step
shifts per-prompt-local offsets into the combined batch arrays and
patches single-request leaves' ``KV_len`` to the current partial-
last-page count.

Cache key per prompt follows the bs_kernel pattern in
``methods/bs_kernel/driver.py:284``:
``(id(pages_prefix), len, ((id, len) per beam's pages_tail))``. List
identity catches forks (page_driver copies ``parent.pages_tail`` to a
fresh list at ``page_driver.py:726``); length catches page-boundary
appends. An earlier version used a single batch-wide structural key
``(len(tree_info), B, K, total_pages)`` (~36% hit rate at B=32 — any
single prompt's fork invalidated all of them); per-prompt caching
brings hit rate to >90% on cells with frequent forks. Disable for
A/B with ``BE_DEFT_PLAN_CACHE=0``.
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
    _build_radix_tree_pages_single,
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
class _DeftPromptCacheEntry:
    """Cached per-prompt sub-tree arrays. All offsets are PROMPT-LOCAL;
    ``_concat_and_h2d`` adds cumulative shifts to convert them to
    batch-wide offsets at concat time. Leaf KV_lens are mutated in
    place each step before concat (idempotent overwrite — every step
    re-patches all of the prompt's leaf entries).

    KVMapQ_List stores GLOBAL beam indices (``b*K + local_k``) so the
    arrays can be concatenated as-is.
    """
    KV_indices_np: np.ndarray
    KV_indices_offset_np: np.ndarray   # LOCAL
    KV_len_np: np.ndarray              # mutable: leaf entries patched per step
    KVMapQ_List_np: np.ndarray         # global rids
    KVMapQ_List_Offset_np: np.ndarray  # LOCAL
    KVMapQ_List_Len_np: np.ndarray
    # Patch info for this prompt's single-request leaf entries:
    leaf_entry_idxs: np.ndarray        # LOCAL entry indices into KV_len_np
    leaf_entry_base: np.ndarray        # (n_pages - 1) * page_size per leaf
    b_owner: int                       # owning prompt — constant for this entry


def _build_prompt_cache_key(prefix: list[int], tails: list[list[int]]) -> tuple:
    """bs_kernel-style per-prompt key. id() captures forks (fresh
    pages_tail lists at page_driver.py:726); len() captures page-
    boundary appends and CoW-driven topology shifts within the
    prompt."""
    return (id(prefix), len(prefix), tuple((id(t), len(t)) for t in tails))


def _pack_deft_arrays_for_prompt(
    prefix: list[int],
    tails: list[list[int]],
    K: int,
    page_size: int,
    b: int,
) -> _DeftPromptCacheEntry:
    """Build one prompt's radix → slot-expansion → DeFT-format arrays.

    Empty ``leaf_partial_last`` passed to ``_expand_pages_to_slots`` →
    leaves get full-page slot ranges; ``_concat_and_h2d`` then writes
    the correct partial-count to KV_len per step. The kernel masks
    beyond KV_len so unwritten slots are never read.
    """
    tree_info, node_pages = _build_radix_tree_pages_single(prefix, tails)
    node_slots = _expand_pages_to_slots(tree_info, node_pages, page_size, {})

    kv_indices_chunks: list[np.ndarray] = []
    kv_indices_offset: list[int] = []
    kv_len: list[int] = []
    kvmapq_chunks: list[np.ndarray] = []
    kvmapq_offset: list[int] = []
    kvmapq_len: list[int] = []
    leaf_entry_idxs: list[int] = []
    leaf_entry_base: list[int] = []

    kv_acc = 0
    mapq_acc = 0
    entry_idx = 0
    base_rid = b * K  # local request ids [0, K) → global [b*K, b*K + K)

    for i, n in enumerate(tree_info):
        n_slots = node_slots[i].size
        reqs = n.requests
        if n_slots == 0 or not reqs:
            continue
        kv_start = kv_acc
        kv_indices_chunks.append(node_slots[i])
        kv_acc += n_slots
        reqs_arr = (np.asarray(reqs, dtype=np.int32) + base_rid)
        n_reqs = reqs_arr.size

        is_single_leaf = (
            n.num_children == 0 and n_reqs == 1 and len(node_pages[i]) > 0
        )
        if is_single_leaf:
            base_len = (len(node_pages[i]) - 1) * page_size

        for off in range(0, n_reqs, _DEFT_BLOCK_M):
            chunk = reqs_arr[off : off + _DEFT_BLOCK_M]
            kvmapq_chunks.append(chunk)
            kvmapq_offset.append(mapq_acc)
            kvmapq_len.append(int(chunk.size))
            mapq_acc += chunk.size
            kv_indices_offset.append(kv_start)
            kv_len.append(n_slots)
            if is_single_leaf:
                leaf_entry_idxs.append(entry_idx)
                leaf_entry_base.append(base_len)
            entry_idx += 1

    return _DeftPromptCacheEntry(
        KV_indices_np=(np.concatenate(kv_indices_chunks)
                       if kv_indices_chunks else np.empty(0, dtype=np.int32)),
        KV_indices_offset_np=np.asarray(kv_indices_offset, dtype=np.int32),
        KV_len_np=np.asarray(kv_len, dtype=np.int32),
        KVMapQ_List_np=(np.concatenate(kvmapq_chunks)
                        if kvmapq_chunks else np.empty(0, dtype=np.int32)),
        KVMapQ_List_Offset_np=np.asarray(kvmapq_offset, dtype=np.int32),
        KVMapQ_List_Len_np=np.asarray(kvmapq_len, dtype=np.int32),
        leaf_entry_idxs=np.asarray(leaf_entry_idxs, dtype=np.int32),
        leaf_entry_base=np.asarray(leaf_entry_base, dtype=np.int32),
        b_owner=b,
    )


def _concat_and_h2d(
    per_prompt: list[_DeftPromptCacheEntry],
    current_pos: list[int],
    page_size: int,
    device: torch.device,
) -> _DeftMeta:
    """Patch each per-prompt entry's leaf KV_lens, then concatenate B
    sets with offset shifts and do a single H2D copy.

    Per-step cost on the cache-hit path: O(B + total_leaf_entries) for
    patching + O(B * n_entries_per_prompt) for the offset shifts —
    avoids the entire radix walk + slot expansion + Python pack loop.
    """
    # 1) Patch leaf KV_lens in each per-prompt entry (mutates in place).
    for entry in per_prompt:
        if entry.leaf_entry_idxs.size > 0:
            partial = current_pos[entry.b_owner] % page_size + 1
            entry.KV_len_np[entry.leaf_entry_idxs] = (
                entry.leaf_entry_base + partial
            )

    # 2) Concatenate B sets with cumulative offset shifts on
    #    KV_indices_offset / KVMapQ_List_Offset (everything else stacks
    #    as-is since KV_indices entries are absolute slot indices and
    #    KVMapQ_List entries are absolute beam ids).
    kv_indices_chunks: list[np.ndarray] = []
    kv_offsets_shifted: list[np.ndarray] = []
    kv_len_chunks: list[np.ndarray] = []
    kvmapq_chunks: list[np.ndarray] = []
    kvmapq_offsets_shifted: list[np.ndarray] = []
    kvmapq_len_chunks: list[np.ndarray] = []
    kv_acc = 0
    mapq_acc = 0
    for e in per_prompt:
        kv_indices_chunks.append(e.KV_indices_np)
        kv_offsets_shifted.append(e.KV_indices_offset_np + kv_acc)
        kv_len_chunks.append(e.KV_len_np)
        kvmapq_chunks.append(e.KVMapQ_List_np)
        kvmapq_offsets_shifted.append(e.KVMapQ_List_Offset_np + mapq_acc)
        kvmapq_len_chunks.append(e.KVMapQ_List_Len_np)
        kv_acc += e.KV_indices_np.size
        mapq_acc += e.KVMapQ_List_np.size

    def _cat(chunks):
        if chunks:
            return np.concatenate(chunks)
        return np.empty(0, dtype=np.int32)

    arrays = (
        _cat(kv_indices_chunks),         # 0
        _cat(kv_offsets_shifted),        # 1
        _cat(kv_len_chunks),             # 2
        _cat(kvmapq_chunks),             # 3
        _cat(kvmapq_offsets_shifted),    # 4
        _cat(kvmapq_len_chunks),         # 5
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
    __slots__ = ("out_buf", "plan_cache_keys", "plan_cache_entries")


@dataclass
class DeftBackend:
    name: str = "deft"
    use_split_pages: bool = True

    _plan_trace: list = field(default_factory=list, init=False, repr=False)
    # Cache hit/miss counters at the *per-prompt* granularity (each
    # decode step contributes B lookups). Visible via the backend
    # handle for benches that want to print them.
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
        # Per-prompt cache: dict[b] -> (key, entry). Survives across
        # decode steps; stale entries (e.g. from prior beam_search
        # calls with different B) look like misses, harmless.
        wb.plan_cache_keys = {}
        wb.plan_cache_entries = {}
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

        # ---- Per-prompt cache lookup + selective rebuild.
        per_prompt: list[_DeftPromptCacheEntry] = []
        for b in range(B):
            bp_b = beams_per_prompt[b]
            prefix = bp_b[0].pages_prefix
            tails = [beam.pages_tail for beam in bp_b]

            if cache_enabled:
                key = _build_prompt_cache_key(prefix, tails)
                cached_key = wrappers.plan_cache_keys.get(b)
                if cached_key == key:
                    per_prompt.append(wrappers.plan_cache_entries[b])
                    self._plan_hits += 1
                    continue
            else:
                key = None

            self._plan_misses += 1
            entry = _pack_deft_arrays_for_prompt(prefix, tails, K, ps, b)
            if cache_enabled:
                wrappers.plan_cache_keys[b] = key
                wrappers.plan_cache_entries[b] = entry
            per_prompt.append(entry)

        meta = _concat_and_h2d(per_prompt, current_pos, ps, device)

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

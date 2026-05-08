"""Baseline — flashinfer's non-fused MultiLevelCascadeAttentionWrapper.

This file is a near-clone of methods/adaptive_pool.py with one
substitution: ``FusedMultiLevelCascadeAttentionWrapper`` is replaced
with the older ``MultiLevelCascadeAttentionWrapper``. Both wrappers
express the same 2-pool / 3-level cascade computation. The fused
variant (used by adaptive_pool and bs_kernel) collapses every per-level
attention launch and the LSE merges into a single kernel launch; the
non-fused wrapper runs ``2L − 1`` launches (one prefill per level plus
``L − 1`` LSE merges).

Comparing this baseline against ``adaptive_pool`` and ``bs_kernel``
isolates the cost of kernel-launch overhead and the LSE-merge round
trips on top of the same cascade computation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import flashinfer.page
import torch
import torch.nn.functional as F
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
)

from ..decoding import (
    DecodeSelect,
    PrefillSelect,
    standard_decode_select,
    standard_prefill_select,
)
from ..methods.adaptive_pool import _pack_batched_cascade_arrays
from ..models.attention import AttentionContext
from ..page_table import PageTable


# ---------------------------------------------------------------------------
# Attention contexts
# ---------------------------------------------------------------------------


@dataclass
class _PrefillCtx(AttentionContext):
    """Standard ragged paged prefill (one request per prompt)."""
    page_table: PageTable
    kv_page_indices: torch.Tensor
    kv_page_offsets: torch.Tensor
    wrapper: BatchPrefillWithPagedKVCacheWrapper
    _write_helper_indptr: torch.Tensor | None = field(default=None, repr=False)

    def attend(self, q, k, v, layer_idx):
        num_kv_heads = self.page_table.head_num
        head_dim = self.page_table.head_dim
        num_heads = q.shape[-1] // head_dim
        kv_cache = self.page_table.kv_cache_at_layer[layer_idx]

        k_3d = k.view(-1, num_kv_heads, head_dim)
        v_3d = v.view(-1, num_kv_heads, head_dim)
        nnz = k_3d.shape[0]
        device = k_3d.device
        if (
            self._write_helper_indptr is None
            or self._write_helper_indptr.shape[0] < nnz + 1
        ):
            self._write_helper_indptr = torch.arange(
                nnz + 1, dtype=torch.int32, device=device,
            )
        batch_idx = self._write_helper_indptr[:nnz]
        kv_indptr = self._write_helper_indptr[: nnz + 1]
        flashinfer.page.append_paged_kv_cache(
            append_key=k_3d,
            append_value=v_3d,
            batch_indices=batch_idx,
            positions=self.kv_page_offsets,
            paged_kv_cache=kv_cache,
            kv_indices=self.kv_page_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=self.kv_page_offsets,
            kv_layout="NHD",
        )
        q_3d = q.view(-1, num_heads, head_dim)
        out = self.wrapper.run(q_3d, kv_cache)
        return out.reshape(*q.shape[:-1], num_heads * head_dim)


@dataclass
class AdaptivePoolContext(AttentionContext):
    """Per-decode-step state: writes new K/V then runs the fused cascade.

    The cascade wrapper has been pre-planned with the per-step (qo_indptr,
    paged_kv_indptr, paged_kv_indices, last_page_len) arrays for L levels.
    """
    page_table: PageTable
    write_pi: torch.Tensor       # [K] int32 — page idx per beam to write into
    write_po: torch.Tensor       # [K] int32 — offset within page per beam
    wrapper: MultiLevelCascadeAttentionWrapper
    _write_helper_indptr: torch.Tensor | None = field(default=None, repr=False)

    def attend(self, q, k, v, layer_idx):
        num_kv_heads = self.page_table.head_num
        head_dim = self.page_table.head_dim
        num_heads = q.shape[-1] // head_dim
        kv_cache = self.page_table.kv_cache_at_layer[layer_idx]

        k_3d = k.view(-1, num_kv_heads, head_dim)
        v_3d = v.view(-1, num_kv_heads, head_dim)
        nnz = k_3d.shape[0]
        device = k_3d.device
        if (
            self._write_helper_indptr is None
            or self._write_helper_indptr.shape[0] < nnz + 1
        ):
            self._write_helper_indptr = torch.arange(
                nnz + 1, dtype=torch.int32, device=device,
            )
        batch_idx = self._write_helper_indptr[:nnz]
        kv_indptr = self._write_helper_indptr[: nnz + 1]
        flashinfer.page.append_paged_kv_cache(
            append_key=k_3d,
            append_value=v_3d,
            batch_indices=batch_idx,
            positions=self.write_po,
            paged_kv_cache=kv_cache,
            kv_indices=self.write_pi,
            kv_indptr=kv_indptr,
            kv_last_page_len=self.write_po,
            kv_layout="NHD",
        )

        q_3d = q.view(-1, num_heads, head_dim)
        out = self.wrapper.run(q_3d, kv_cache)
        return out.reshape(*q.shape[:-1], num_heads * head_dim)


# ---------------------------------------------------------------------------
# Beam state + page-refcount helpers (shared with paged.py's idea)
# ---------------------------------------------------------------------------


@dataclass
class Beam:
    token_ids: list[int]
    cum_log_prob: float
    pages: list[int]


def _add_ref(rc: dict[int, int], page: int, count: int = 1) -> None:
    rc[page] = rc.get(page, 0) + count


def _remove_ref(
    rc: dict[int, int], page: int, page_table: PageTable, count: int = 1,
) -> None:
    rc[page] -= count
    assert rc[page] >= 0
    if rc[page] == 0:
        page_table.free_block(page)
        del rc[page]


# ---------------------------------------------------------------------------
# Fixed 2-level layout — vanilla cascade baseline.
#
# Adaptive 2/3-level switching is the novelty of bs_kernel / adaptive_pool;
# this baseline must NOT use it. The layout here is always the natural
# 2-level decomposition the tree dictates: shared LCA prefix + per-beam
# unique tails. The LCA depth depends on tree shape; the level count does
# not.
# ---------------------------------------------------------------------------


def _two_level_layout(
    pages_per_beam: list[list[int]],
    last_page_len_per_beam: list[int],
    K: int,
    *,
    start_lca: int = 0,
):
    """Always-2-level cascade layout.

    Level 0: longest common page-prefix across all K beams (1 group of K).
    Level 1: each beam's unique tail past the LCA (K singleton groups).

    No heuristic, no group-uniformity detection, no intermediate level.

    Returns ``(levels, lca)``. The optional ``start_lca`` lets callers
    seed the LCA scan from a value cached at the previous decode step
    (LCA is provably monotone non-decreasing across steps).
    """
    # LCA depth — longest common page prefix. Resume from the cached
    # value when the caller has one.
    lca = start_lca
    min_len = min(len(p) for p in pages_per_beam)
    while lca < min_len:
        first = pages_per_beam[0][lca]
        if all(p[lca] == first for p in pages_per_beam):
            lca += 1
        else:
            break

    shared_pages = pages_per_beam[0][:lca]
    # `page_size` sentinel — the caller substitutes in the actual page size
    # for shared pages (they're always fully-populated past pages: if any
    # beam had appended into them mid-page they wouldn't be shared, since
    # CoW would have split).
    page_size = -1

    levels: list[tuple[list[int], list[list[int]], list[int]]] = []
    if shared_pages:
        levels.append(([K], [shared_pages], [page_size]))
    per_beam_tail = [pages_per_beam[i][lca:] for i in range(K)]
    per_beam_lpl = list(last_page_len_per_beam)
    levels.append(([1] * K, per_beam_tail, per_beam_lpl))
    return levels, lca


def _build_cascade_plan(
    pages_per_beam: list[list[int]],
    last_page_len_per_beam: list[int],
    page_size: int,
    K: int,
    *,
    device: torch.device,
):
    """Pack the fixed 2-level (qo_indptr, kv_indptr, kv_indices, last_page_len)
    arrays. Returns the four arrays. Beam order is the identity (no
    reordering — that's only needed for the 3-level intermediate path).
    """
    levels, _lca = _two_level_layout(pages_per_beam, last_page_len_per_beam, K)
    # Substitute the page_size sentinel introduced inside _adaptive_levels.
    qo_arr: list[torch.Tensor] = []
    kvp_arr: list[torch.Tensor] = []
    kvi_arr: list[torch.Tensor] = []
    kvl_arr: list[torch.Tensor] = []

    # Track total qo rows after potential row-reordering (= K).
    for sizes, group_pages, group_lpl in levels:
        qo_indptr = [0]
        for s in sizes:
            qo_indptr.append(qo_indptr[-1] + s)
        kv_indptr = [0]
        kv_indices: list[int] = []
        last_page = []
        for pages, lpl in zip(group_pages, group_lpl):
            kv_indices.extend(pages)
            kv_indptr.append(kv_indptr[-1] + len(pages))
            # sentinel -1 → page_size (shared/intermediate; fully-populated
            # earlier page); else use the actual per-beam lpl.
            last_page.append(page_size if lpl == -1 else lpl)
        qo_arr.append(torch.tensor(qo_indptr, dtype=torch.int32, device=device))
        kvp_arr.append(torch.tensor(kv_indptr, dtype=torch.int32, device=device))
        kvi_arr.append(torch.tensor(kv_indices, dtype=torch.int32, device=device))
        kvl_arr.append(torch.tensor(last_page, dtype=torch.int32, device=device))
    return qo_arr, kvp_arr, kvi_arr, kvl_arr


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


from dataclasses import dataclass as _dataclass


class _MlcaWrappers:
    __slots__ = ("cascade",)


@_dataclass
class MlcaBackend:
    name: str = "mlca"
    use_split_pages: bool = True

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
    ):
        wb = _MlcaWrappers()
        # Non-fused cascade — runs (2L-1) launches per step (L attn + L-1
        # merges) where L=2: 3 launches per step total.
        wb.cascade = MultiLevelCascadeAttentionWrapper(
            num_levels=2,
            float_workspace_buffer=workspace_buffer,
            kv_layout="NHD",
        )
        return wb

    def plan_decode_step(
        self,
        *,
        wrappers,
        beams_per_prompt,
        current_pos,
        page_table,
        K,
        B,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
        device,
        last_lca_per_prompt,
    ):
        from ..page_driver import StepPlan

        ps = page_size
        levels_per_prompt = []
        for b in range(B):
            pos = current_pos[b]
            off = pos % ps
            bp_b = beams_per_prompt[b]
            # _two_level_layout takes the unified pages list, but split-form
            # beams have prefix + tail. Build the unified list lazily here;
            # the LCA scan resumes from start_lca so the cost is bounded by
            # the small per-beam tail rather than the full prefix.
            pages_per_beam = [
                bm.pages_prefix + bm.pages_tail for bm in bp_b
            ]
            lpl_per_beam = [off + 1] * K
            levels, lca_b = _two_level_layout(
                pages_per_beam, lpl_per_beam, K,
                start_lca=last_lca_per_prompt[b],
            )
            last_lca_per_prompt[b] = lca_b
            levels_per_prompt.append(levels)

        qo_arr, kvp_arr, kvi_arr, kvl_arr = _pack_batched_cascade_arrays(
            levels_per_prompt, ps, device,
        )
        wrappers.cascade.plan(
            qo_indptr_arr=qo_arr,
            paged_kv_indptr_arr=kvp_arr,
            paged_kv_indices_arr=kvi_arr,
            paged_kv_last_page_len=kvl_arr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=ps,
            causal=False,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

        write_pi_list = []
        write_po_list = []
        for b in range(B):
            pos = current_pos[b]
            off = pos % ps
            pli = pos // ps
            bp_b = beams_per_prompt[b]
            prefix_len = len(bp_b[0].pages_prefix)
            tail_idx = pli - prefix_len
            for beam in bp_b:
                write_pi_list.append(beam.pages_tail[tail_idx])
                write_po_list.append(off)
        ctx = AdaptivePoolContext(
            page_table=page_table,
            write_pi=torch.tensor(
                write_pi_list, dtype=torch.int32, device=device),
            write_po=torch.tensor(
                write_po_list, dtype=torch.int32, device=device),
            wrapper=wrappers.cascade,
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
    select_at_prefill: PrefillSelect = standard_prefill_select,
    select_at_decode: DecodeSelect = standard_decode_select,
):
    """Vanilla cascade beam search via flashinfer's non-fused
    ``MultiLevelCascadeAttentionWrapper``.

    Always 2 cascade levels: shared LCA prefix (level 0) + per-beam unique
    tails (level 1). The LCA depth is read off the tree each step; the
    *level count* is fixed. No 2/3-level adaptive switching — that's the
    novelty of ``adaptive_pool`` / ``bs_kernel`` and not part of this
    baseline.

    The non-fused wrapper runs 2L−1 = 3 kernel launches per step (one
    attention launch per level + one LSE merge); ``adaptive_pool`` uses
    the fused wrapper that collapses the 3 launches into 1.
    """
    from ..page_driver import beam_search as _shared_beam_search

    return _shared_beam_search(
        model, config, prompt_ids, max_new_tokens, beam_width,
        backend=MlcaBackend(),
        page_size=page_size,
        max_num_pages=max_num_pages,
        device=device,
        dtype=dtype,
        return_timings=return_timings,
        select_at_prefill=select_at_prefill,
        select_at_decode=select_at_decode,
    )

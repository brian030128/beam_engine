"""bs_kernel beam-search driver.

Thin wrapper around the shared page-table driver in
``beam_engine.page_driver``. The interesting per-method logic lives in
``BsKernelBackend.plan_decode_step``: it builds per-prompt workloads,
calls ``cost_model.pick_strategy_batch`` to choose
(strategy, depth, pool_count, t_large), then dispatches to one of:

  * PER_BEAM           → ``BatchDecodeWithPagedKVCacheWrapper``
  * SHARED_*_1POOL     → fused cascade wrapper (single launch)
  * SHARED_*_2POOL     → non-fused MLCA-style wrapper (3 or 5 launches)

Wrapper instances are constructed once at startup and reused across
decode steps.
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    FusedMultiLevelCascadeAttentionWrapper,
    MultiLevelCascadeAttentionWrapper,
)

from ...baselines.paged import PagedAttentionContext
from ...decoding import (
    DecodeSelect,
    PrefillSelect,
    standard_decode_select,
    standard_prefill_select,
)
from ...page_driver import StepPlan, WrapperBundle, beam_search as _shared_beam_search
from ...page_table import PageTable
from ..adaptive_pool import (
    AdaptivePoolContext,
    Beam,
    _adaptive_levels,
)
from .calibrate import load_or_defaults
from .cost_model import (
    Coefficients,
    IntermediateShape,
    Pick,
    Strategy,
    WorkloadShape,
    pick_strategy,
    pick_strategy_batch,
)
from .decode_tail_context import DecodeTailCascadeContext


# ---------------------------------------------------------------------------
# Layout helpers (kept here because they're bs_kernel-specific:
# adaptive_pool always uses depth=2 in batched mode, so it doesn't need
# d3-aware machinery).
# ---------------------------------------------------------------------------


def _collapse_levels(levels: list, target_depth: int) -> list:
    """Collapse a deeper cascade layout to ``target_depth`` in O(K).

    ``_adaptive_levels(max_levels=D_max)`` may emit any number of levels
    in ``[1..D_max]``. The picker chooses ``pick.depth ∈ {2..D_max}``;
    when the natural depth exceeds ``target_depth`` we merge the
    *deepest* intermediate levels into the per-beam tail (preserving
    the shallowest intermediates, which are closer to the LCA and thus
    have larger group sizes and bigger sharing benefit).

    Specifically: keep ``levels[0..target_depth-2]`` (root + the
    shallowest ``target_depth-2`` intermediates), and merge
    ``levels[target_depth-1..N-2]`` (deeper intermediates) into
    ``levels[N-1]`` (per-beam tail) by prepending their pages onto each
    beam's tail in group order.

    Cost-model equivalence: ``_per_prompt_levels(w, depth=target)``
    folds the dropped intermediate pages back into each beam's suffix
    length, which is exactly what this function does at the page-list
    level.
    """
    if len(levels) <= target_depth:
        return levels
    if target_depth < 2:
        raise ValueError(f"target_depth must be >= 2, got {target_depth}")

    # Levels we keep verbatim: root + shallowest (target_depth - 2)
    # intermediates.
    kept = levels[: target_depth - 1]
    # Levels we merge into the per-beam tail.
    to_merge = levels[target_depth - 1 : -1]
    per_beam = levels[-1]
    pb_sizes, pb_pages, pb_lpl = per_beam

    # For each beam, the merged tail = concat of (its pages from each
    # to_merge level, in shallowest→deepest order) + its original
    # per-beam tail. Each to_merge level is structured as
    # (group_sizes, group_pages, group_lpl) and the per-beam mapping
    # follows group expansion order, which matches the per-beam tail
    # ordering downstream.
    K = len(pb_pages)
    prepend_per_beam: list[list[int]] = [[] for _ in range(K)]
    for sizes, group_pages, _lpl in to_merge:
        beam_idx = 0
        for gi, gsize in enumerate(sizes):
            grp_pages = group_pages[gi]
            for _ in range(gsize):
                prepend_per_beam[beam_idx].extend(grp_pages)
                beam_idx += 1
        # Sanity: every level in to_merge must produce the same K beams
        # in the same order (true by construction in _adaptive_levels).
    merged_tail = [prepend_per_beam[i] + pb_pages[i] for i in range(K)]
    return kept + [(pb_sizes, merged_tail, pb_lpl)]


def _collapse_d3_to_d2(levels: list) -> list:
    """Backwards-compat shim — see :func:`_collapse_levels`."""
    return _collapse_levels(levels, 2)


def _workload_from_levels(
    levels: list,
    K: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    num_qo_heads: int = 0,
) -> WorkloadShape:
    """Derive the cost model's WorkloadShape from `_adaptive_levels`'s
    output. Generalized to support N levels (root + N-2 intermediates +
    per-beam) for ``N >= 2``.

    The returned ``IntermediateShape.levels`` has one entry per
    intermediate cascade level, each entry a list of
    ``(group_size, group_kv_tokens)`` tuples in group order.
    """
    has_shared = len(levels) >= 2
    if has_shared:
        _sizes_0, pages_0, _lpl_0 = levels[0]
        L_p = len(pages_0[0]) * page_size
    else:
        L_p = 0

    sizes_last, pages_last, lpl_last = levels[-1]
    suffix_lens: list[int] = []
    for i in range(K):
        if i < len(pages_last):
            tail_pages = pages_last[i]
            tail_lpl = lpl_last[i]
            tail_tokens = (
                (len(tail_pages) - 1) * page_size + tail_lpl
                if tail_pages else 0
            )
            suffix_lens.append(tail_tokens)
        else:
            suffix_lens.append(0)

    intermediate: IntermediateShape | None = None
    if len(levels) >= 3:
        # All levels except the root and the per-beam tail are
        # intermediates. Each contributes a list of (group_size,
        # group_kv_tokens) tuples to IntermediateShape.levels.
        intermediate_levels: list[list[tuple[int, int]]] = []
        for li in range(1, len(levels) - 1):
            sizes_mid, pages_mid, _ = levels[li]
            groups = [
                (sizes_mid[gi], len(pages_mid[gi]) * page_size)
                for gi in range(len(sizes_mid))
            ]
            intermediate_levels.append(groups)
        intermediate = IntermediateShape(levels=intermediate_levels)

    bytes_per_kv = 2 * num_kv_heads * head_dim * dtype_bytes
    return WorkloadShape(
        K=K,
        L_p=L_p,
        suffix_lens=suffix_lens,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_kv=bytes_per_kv,
        intermediate=intermediate,
        num_qo_heads=num_qo_heads,
    )


def _pack_batched_cascade_arrays_any_depth(
    levels_per_prompt: list[list],
    page_size: int,
    device: torch.device,
    *,
    n_levels: int,
    skip_levels: tuple = (),
):
    """Generalized batched-cascade-array packer for any ``n_levels >= 2``.

    Every prompt is expected to contribute exactly ``n_levels`` levels;
    the picker only emits depth=N when every prompt's adaptive layout
    can support N levels, so this invariant holds at the call site.

    ``skip_levels`` — set of level indices to omit from the build (used
    when the caller has cached GPU tensors for those levels from a
    previous step and wants to skip the Python list extends + H2D copy
    at the dominant level. The returned per-level lists have ``None`` at
    skipped indices; the caller is responsible for substituting the
    cached tensors back in.
    """
    qo_arr_per_level: list[list[int]] = [[0] for _ in range(n_levels)]
    kvp_arr_per_level: list[list[int]] = [[0] for _ in range(n_levels)]
    kvi_arr_per_level: list[list[int]] = [[] for _ in range(n_levels)]
    kvl_arr_per_level: list[list[int]] = [[] for _ in range(n_levels)]

    for levels in levels_per_prompt:
        assert len(levels) == n_levels, (
            f"_pack_batched_cascade_arrays_any_depth: prompt has "
            f"{len(levels)} levels, expected {n_levels}"
        )
        for li, (sizes, group_pages, group_lpl) in enumerate(levels):
            if li in skip_levels:
                continue
            for s in sizes:
                qo_arr_per_level[li].append(qo_arr_per_level[li][-1] + s)
            for pages, lpl in zip(group_pages, group_lpl):
                kvi_arr_per_level[li].extend(pages)
                kvp_arr_per_level[li].append(
                    kvp_arr_per_level[li][-1] + len(pages))
                kvl_arr_per_level[li].append(
                    page_size if lpl == -1 else lpl)

    def _maybe(i, lst):
        if i in skip_levels:
            return None
        return torch.tensor(lst, dtype=torch.int32, device=device)

    qo_arr = [_maybe(i, qo_arr_per_level[i]) for i in range(n_levels)]
    kvp_arr = [_maybe(i, kvp_arr_per_level[i]) for i in range(n_levels)]
    kvi_arr = [_maybe(i, kvi_arr_per_level[i]) for i in range(n_levels)]
    kvl_arr = [_maybe(i, kvl_arr_per_level[i]) for i in range(n_levels)]
    # Also return the host-side cumsum lists used to build qo_arr / kvp_arr.
    # The cascade plan-state cache uses these as a signature: schedule is
    # purely a function of per-level qo_indptr_h + kv_indptr_h, so two
    # steps with identical lists share an identical schedule.
    return qo_arr, kvp_arr, kvi_arr, kvl_arr, qo_arr_per_level, kvp_arr_per_level


# ---------------------------------------------------------------------------
# Decomp/workload cache
# ---------------------------------------------------------------------------


@dataclass
class _DecompCacheEntry:
    """Cacheable part of one prompt's ``_adaptive_levels`` +
    ``_workload_from_levels`` output.

    Excludes ``last_page_len`` — that's per-step (``off + 1``) and is
    patched in on every hit. Includes everything that's a pure function
    of the page-list topology.

    The cache key is a tuple of (id, len) tuples over pages_prefix and
    every beam's pages_tail. List identity changes only on fork (a
    parent's tail is copied to a fresh list, ``page_driver.py:680``);
    list length changes only on page-boundary append (``off == 0``).
    Within a 16-step page epoch with no fork, the key is invariant
    and every step hits.
    """
    # The level structure with placeholder lpl entries. On cache hit we
    # patch the leaf-level lpl in place before returning (this list is
    # rebuilt fresh for the caller, so no aliasing into the caller's
    # downstream state).
    sizes_per_level: list                      # one list[int] per level
    pages_per_level: list                      # one list[list[int]] per level
    beam_order: list                           # length-K permutation
    lca_b: int
    # Cached WorkloadShape inputs that don't depend on lpl.
    L_p: int
    intermediate: object                       # IntermediateShape | None
    # Per-leaf tail page-counts (for fast suffix_lens rebuild).
    leaf_pages_lens: list


def _build_decomp_key(
    pages_prefix: list,
    pages_tails: list,
    max_depth: int,
) -> tuple:
    """One key tuple per prompt. Order-sensitive over beams (cache miss
    if beams reorder, which doesn't happen in our drivers but would be
    visible in the key)."""
    return (
        id(pages_prefix),
        len(pages_prefix),
        tuple((id(t), len(t)) for t in pages_tails),
        max_depth,
    )


def _materialize_levels_from_cache(
    entry: _DecompCacheEntry,
    lpl: int,
    page_size_sentinel: int = -1,
) -> list:
    """Build the levels list the dispatch code expects from a cached
    entry. ``lpl`` is the current ``off + 1`` for this prompt — applies
    to the leaf level only; intermediates use the ``page_size`` sentinel
    (``-1``) which downstream substitutes with the true page size."""
    levels = []
    n_levels = len(entry.sizes_per_level)
    for li in range(n_levels):
        sizes = entry.sizes_per_level[li]
        pages = entry.pages_per_level[li]
        if li == n_levels - 1:
            # Per-beam tail level: K entries of single-token lpls.
            lpls = [lpl] * len(sizes)
        else:
            # Non-leaf level: page_size sentinel.
            lpls = [page_size_sentinel] * len(sizes)
        levels.append((sizes, pages, lpls))
    return levels


def _workload_from_cache(
    entry: _DecompCacheEntry,
    K: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    lpl: int,
    num_qo_heads: int = 0,
) -> WorkloadShape:
    """Rebuild the ``WorkloadShape`` from cached structure + current
    lpl. Equivalent to calling ``_workload_from_levels`` on the
    materialised levels but skips the level-walk; suffix_lens is the
    only field that depends on ``lpl``."""
    suffix_lens = [
        ((entry.leaf_pages_lens[i] - 1) * page_size + lpl)
        if entry.leaf_pages_lens[i] > 0 else 0
        for i in range(K)
    ]
    return WorkloadShape(
        K=K,
        L_p=entry.L_p,
        suffix_lens=suffix_lens,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_kv=2 * num_kv_heads * head_dim * dtype_bytes,
        intermediate=entry.intermediate,
        num_qo_heads=num_qo_heads,
    )


def _entry_from_decomp(
    levels: list,
    beam_order: list,
    lca_b: int,
    K: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    num_qo_heads: int = 0,
) -> tuple[_DecompCacheEntry, WorkloadShape]:
    """Build the cache entry from a fresh ``_adaptive_levels`` output
    (on cache miss). Also returns the current-step WorkloadShape so the
    miss path doesn't pay a second pass."""
    # Compute WorkloadShape via the existing helper (this also derives
    # L_p and intermediate, which we need for the entry).
    w = _workload_from_levels(
        levels, K, page_size, num_kv_heads, head_dim, dtype_bytes,
        num_qo_heads=num_qo_heads,
    )
    # Tease apart the levels into (sizes, pages, lpl) parallel lists.
    sizes_per_level = [lv[0] for lv in levels]
    pages_per_level = [lv[1] for lv in levels]
    leaf_pages = levels[-1][1]
    leaf_pages_lens = [len(leaf_pages[i]) if i < len(leaf_pages) else 0
                       for i in range(K)]
    entry = _DecompCacheEntry(
        sizes_per_level=sizes_per_level,
        pages_per_level=pages_per_level,
        beam_order=beam_order,
        lca_b=lca_b,
        L_p=w.L_p,
        intermediate=w.intermediate,
        leaf_pages_lens=leaf_pages_lens,
    )
    return entry, w


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class _BsKernelWrappers(WrapperBundle):
    """Wrapper bundle for bs_kernel. Wrappers cached at startup; the
    picker chooses one per step.

    Cascade wrappers are indexed by depth: ``cascade_wrappers[d]`` is a
    ``FusedMultiLevelCascadeAttentionWrapper`` configured for ``d``
    levels. All share the same ``max_levels`` template parameter at JIT
    time, so they share a single compiled kernel binary. Same pattern
    for the non-fused MLCA path (``cascade_dual_wrappers[d]``).

    For DEC_TAIL strategies the prefix + (optional) intermediate levels
    go through a single ``FusedMultiLevelCascadeAttentionWrapper`` per
    dispatch depth (``dec_tail_cascade_wrappers[d]``, ``num_levels=d-1``
    — covers the prefill levels only, excluding the per-beam tail which
    runs through ``decode_wrapper``). Empirically the cascade wrapper
    is 5× faster than per-level ``BatchPrefillWithPagedKVCacheWrapper``
    on long-prefix workloads; see ``docs/cost_model_issues.md`` and
    ``scripts/paper-exp/probe_prefix_wrappers.py``.
    """
    __slots__ = (
        "decode_wrapper",
        "cascade_wrappers",
        "cascade_dual_wrappers",
        "dec_tail_cascade_wrappers",
        "max_depth",
    )


@dataclass
class BsKernelBackend:
    name: str = "bs_kernel"
    use_split_pages: bool = True

    coefficients: Coefficients | None = None
    fused_merge: bool = False
    available_strategies: set[Strategy] | None = None

    # DEC_TAIL plan cache: skip prefix_prefill.plan() when the LCA is
    # unchanged for every prompt across consecutive steps. The LCA is
    # monotone non-decreasing within a beam_search call, so re-plans
    # only happen when a new prefix page becomes shared (bursts at fork
    # events). At K=64/B=32 in steady state, LCA usually grows once
    # every page_size=16 steps → cache hits on ~94% of steps.
    _planned_prefix_lca: list[int] | None = field(
        default=None, init=False, repr=False,
    )

    # Decode-wrapper plan_info cache: FlashInfer's plan() call runs a
    # C++ scheduler that allocates CTAs to (beam, kv-tile) tuples. The
    # scheduling depends on per-beam kv_len, which grows monotonically
    # but only by ±1 token per step. Reuse the plan_info across steps
    # and swap only the (indptr, indices, last_page_len) buffers — the
    # kernel still produces correct outputs (the schedule is suboptimal
    # but not incorrect for our shape regime). Re-plan only when the
    # max per-beam kv_len roughly doubles, which would shift the
    # split-K factor and could exceed workspace.
    _decode_plan_info_cached: object = field(
        default=None, init=False, repr=False,
    )
    _decode_max_kv_at_plan: int = field(
        default=0, init=False, repr=False,
    )

    # SHARED-branch level-0 pack cache: at long L_p the level-0 (prefix)
    # entry in kvi_arr is ~B * lca_pages int32 entries (~15k at K=16/B=8/
    # L_p=30000), and building+H2D dominates shared_pack_ms. Level-0
    # content is determined purely by the LCA prefix pages — invariant
    # while last_lca_per_prompt is unchanged. Cache the 4 packed GPU
    # tensors for level 0 keyed by (LCA tuple, dispatch_depth).
    _shared_l0_cache_key: tuple | None = field(
        default=None, init=False, repr=False,
    )
    _shared_l0_cache_tensors: tuple | None = field(
        default=None, init=False, repr=False,
    )
    # Host-side level-0 cumsums (qo_h, kvp_h) — needed by the plan-state
    # cache signature when level-0 is fetched from the L0 tensor cache
    # (which skips the per-level CPU list build for level 0).
    _shared_l0_cache_h: tuple | None = field(
        default=None, init=False, repr=False,
    )

    # SHARED-branch cascade plan-state cache. FlashInfer's cascade
    # ``wrapper.plan()`` does two ``.cpu()`` roundtrips + a Python
    # scheduler over qo/kv groups + GPU tensor materialization for
    # ``_pool_*_bufs``. Schedule is a pure function of per-level
    # ``qo_indptr_h`` + ``kv_indptr_h``. On steps where these are
    # byte-identical to the previous plan of the same wrapper, the
    # schedule is fully reusable — we only need to overwrite the 4
    # input buffers (concatenated qo_indptr / kv_indptr / kv_indices /
    # last_page_len) on the wrapper and skip plan() entirely.
    # Key: ``id(wrapper)``  →  ``(plan_key, plan_sig)``
    #   plan_key = (depth, pool_count, t_large)
    #   plan_sig = (qo_cumsum_per_level_tuple, kvp_cumsum_per_level_tuple)
    # Hit when both match the current step's values. Miss → full
    # re-plan and overwrite the entry.
    _shared_plan_cache: dict = field(default_factory=dict, init=False, repr=False)

    # Per-prompt decomp/workload cache. ``_adaptive_levels`` builds the
    # cascade level tree by walking each beam's tail page list; at high B
    # (e.g. SGLang multi_chain_reasoning B=32 K=4) that's ~0.25 ms × B
    # of pure Python overhead per step (~7.5 ms total — dominant in
    # plan_decode_step). The result is a pure function of
    # (id+len(pages_prefix), tuple of (id+len) per beam.pages_tail,
    # max_depth); ``last_page_len`` (= off+1) doesn't change the LEVEL
    # STRUCTURE — it only feeds into the leaf-level lpl array and into
    # ``WorkloadShape.suffix_lens``, both of which are cheap to rebuild.
    # So we cache the structural part and patch lpl-dependent fields on
    # every hit. In standard beam search, a fork copies the parent's
    # tail list (``page_driver.py:680``) → new id → cache miss for that
    # prompt; in non-fork cases (steady-state decode, the common case
    # at high K — same regime that drives ``_shared_plan_cache`` to
    # 94% hits) the cache hits within a 16-step page epoch and misses
    # only on page-boundary tail-append steps.
    # Key: prompt index b → ``_DecompCacheEntry`` (one entry per prompt).
    _decomp_cache: dict = field(default_factory=dict, init=False, repr=False)

    # Sub-phase trace for plan_decode_step. Enabled by env var
    # BS_KERNEL_TRACE_PLAN=1. Each entry is a dict of per-phase ms +
    # cache-hit flags for one step.
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
    ) -> _BsKernelWrappers:
        wb = _BsKernelWrappers()
        wb.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout="NHD", use_tensor_cores=True,
        )
        # One fused wrapper per depth ∈ {2..max_cascade_levels}. All
        # share the same ``max_levels`` template parameter, so they
        # share a single JIT-compiled kernel binary; only the runtime
        # ``num_levels`` field on the param struct differs. This lets
        # the picker flip depth between steps with no rebuild cost.
        wb.cascade_wrappers = {}
        wb.cascade_dual_wrappers = {}
        for d in range(2, max_cascade_levels + 1):
            wb.cascade_wrappers[d] = FusedMultiLevelCascadeAttentionWrapper(
                num_levels=d,
                float_workspace_buffer=workspace_buffer,
                kv_layout="NHD",
                device=device,
                max_levels=max_cascade_levels,
            )
            wb.cascade_dual_wrappers[d] = MultiLevelCascadeAttentionWrapper(
                num_levels=d,
                float_workspace_buffer=workspace_buffer,
                kv_layout="NHD",
            )
        wb.max_depth = max_cascade_levels
        # DEC_TAIL prefill: one FusedMultiLevelCascadeAttentionWrapper
        # per dispatch depth, configured for (depth - 1) prefill levels
        # (prefix + intermediates; per-beam tail goes through
        # ``decode_wrapper``). The cascade wrapper is ~5× faster than
        # the per-level ``BatchPrefillWithPagedKVCacheWrapper`` on
        # long-prefix workloads (probe at
        # ``scripts/paper-exp/probe_prefix_wrappers.py``).
        wb.dec_tail_cascade_wrappers = {}
        for d in range(2, max_cascade_levels + 1):
            n_prefill = d - 1
            wb.dec_tail_cascade_wrappers[d] = FusedMultiLevelCascadeAttentionWrapper(
                num_levels=n_prefill,
                float_workspace_buffer=workspace_buffer,
                kv_layout="NHD",
                device=device,
                max_levels=max_cascade_levels,
            )
        return wb

    def plan_decode_step(
        self,
        *,
        wrappers: _BsKernelWrappers,
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
        # KV cache may be quantized (e.g. fp8) while compute stays bf16;
        # the cost model's bytes_per_kv tracks HBM bytes loaded, so use
        # the KV cache's actual element size, not the compute dtype.
        dtype_bytes = torch.tensor(
            [], dtype=page_table.store_dtype,
        ).element_size()
        max_depth = wrappers.max_depth
        trace_on = bool(int(os.environ.get("BS_KERNEL_TRACE_PLAN", "0")))
        # Override fixed_split_size for the DEC_TAIL prefix wrapper plan.
        # Default (empty) leaves FlashInfer's H100 default — which caps
        # padded_batch_size at ~33 with 8 kv-heads, so at K=16/B≥4 the
        # prefix prefill only splits each group into ≤2 kv-chunks. Forcing
        # a smaller kv chunk (= more splits → more CTAs) is the lever we
        # use to investigate the K=16 forward gap vs fasttree's two-stage
        # decode (see medium-K analysis 2026-05-12).
        _split_env = os.environ.get("BS_KERNEL_DEC_TAIL_PREFIX_SPLIT_PAGES", "").strip()
        prefix_fixed_split: int | None = int(_split_env) if _split_env else None
        if trace_on:
            torch.cuda.synchronize()
            _t_start = time.perf_counter()

        # ---- Per-prompt: compute up-to-D_max-level decomposition + workload. ----
        # Fast path: check the per-prompt decomp cache before falling
        # through to ``_adaptive_levels`` + ``_workload_from_levels``.
        # See ``_DecompCacheEntry`` docstring for cache-key rationale.
        # Disable via ``BS_KERNEL_DISABLE_DECOMP_CACHE=1`` for A/B
        # comparison.
        _cache_disabled = bool(int(
            os.environ.get("BS_KERNEL_DISABLE_DECOMP_CACHE", "0")
        ))
        levels_full_per_prompt: list[list] = []
        beam_order_full_per_prompt: list[list[int]] = []
        workloads: list[WorkloadShape] = []
        decomp_hits = 0
        for b in range(B):
            pos = current_pos[b]
            off = pos % ps
            lpl = off + 1
            bp_b = beams_per_prompt[b]
            pages_prefix = bp_b[0].pages_prefix
            pages_tails = [bm.pages_tail for bm in bp_b]
            key = None if _cache_disabled else _build_decomp_key(
                pages_prefix, pages_tails, max_depth,
            )
            cached = (
                None if _cache_disabled else self._decomp_cache.get(b)
            )
            if cached is not None and cached[0] == key:
                entry: _DecompCacheEntry = cached[1]
                levels = _materialize_levels_from_cache(entry, lpl)
                beam_order = entry.beam_order
                lca_b = entry.lca_b
                w = _workload_from_cache(
                    entry, K, ps, num_kv_heads, head_dim, dtype_bytes, lpl,
                    num_qo_heads=num_qo_heads,
                )
                decomp_hits += 1
            else:
                lpl_per_beam = [lpl] * K
                levels, beam_order, lca_b = _adaptive_levels(
                    pages_prefix, pages_tails, lpl_per_beam, K,
                    max_levels=max_depth,
                    start_lca=last_lca_per_prompt[b],
                )
                if not _cache_disabled:
                    entry, w = _entry_from_decomp(
                        levels, beam_order, lca_b, K, ps,
                        num_kv_heads, head_dim, dtype_bytes,
                        num_qo_heads=num_qo_heads,
                    )
                    self._decomp_cache[b] = (key, entry)
                else:
                    w = _workload_from_levels(
                        levels, K, ps, num_kv_heads, head_dim, dtype_bytes,
                        num_qo_heads=num_qo_heads,
                    )
            last_lca_per_prompt[b] = lca_b
            levels_full_per_prompt.append(levels)
            beam_order_full_per_prompt.append(beam_order)
            workloads.append(w)

        if trace_on:
            _t_decomp = time.perf_counter()

        # ---- Cost-model pick across the whole batch. ----
        pick = pick_strategy_batch(
            workloads, self.coefficients,
            fused_merge=self.fused_merge,
            available_strategies=self.available_strategies,
        )
        if trace_on:
            _t_pick = time.perf_counter()

        # ---- Choose dispatch depth + collapse layouts if needed. ----
        dispatch_depth = pick.depth if pick.share else 2
        levels_per_prompt = [
            _collapse_levels(levels_full_per_prompt[b], dispatch_depth)
            for b in range(B)
        ]
        beam_order_per_prompt = beam_order_full_per_prompt

        # ---- Dispatch. ----
        if pick.strategy == Strategy.PER_BEAM:
            # B*K independent paged-decode sequences in one launch.
            # Numpy-bridge build (mirrors paged.plan_decode_step): two
            # passes to size the buffer and slice-fill, then one
            # ``torch.from_numpy().to(device, non_blocking=True)`` per
            # tensor — ~30x faster than ``torch.tensor(list, int32)`` on
            # the multi-thousand-entry flat-indices list at high (B*K).
            n_seqs = B * K
            indptr_np = np.empty(n_seqs + 1, dtype=np.int32)
            indptr_np[0] = 0
            lpl_np = np.empty(n_seqs, dtype=np.int32)
            write_pi_np = np.empty(n_seqs, dtype=np.int32)
            write_po_np = np.empty(n_seqs, dtype=np.int32)
            row = 0
            for b in range(B):
                pos = current_pos[b]
                off = pos % ps
                pli = pos // ps
                bp_b = beams_per_prompt[b]
                prefix_len = len(bp_b[0].pages_prefix)
                tail_idx = pli - prefix_len
                for beam in bp_b:
                    indptr_np[row + 1] = (
                        indptr_np[row]
                        + len(beam.pages_prefix)
                        + len(beam.pages_tail)
                    )
                    lpl_np[row] = off + 1
                    write_pi_np[row] = beam.pages_tail[tail_idx]
                    write_po_np[row] = off
                    row += 1
            total = int(indptr_np[n_seqs])
            indices_np = np.empty(total, dtype=np.int32)
            row = 0
            for b in range(B):
                for beam in beams_per_prompt[b]:
                    start = indptr_np[row]
                    pl = len(beam.pages_prefix)
                    indices_np[start : start + pl] = beam.pages_prefix
                    indices_np[start + pl : indptr_np[row + 1]] = beam.pages_tail
                    row += 1

            if trace_on:
                _t_per_beam_build = time.perf_counter()

            indptr_t = torch.from_numpy(indptr_np).to(device, non_blocking=True)
            indices_t = torch.from_numpy(indices_np).to(device, non_blocking=True)
            lpl_t = torch.from_numpy(lpl_np).to(device, non_blocking=True)
            write_pi_t = torch.from_numpy(write_pi_np).to(device, non_blocking=True)
            write_po_t = torch.from_numpy(write_po_np).to(device, non_blocking=True)
            if trace_on:
                torch.cuda.synchronize()
                _t_per_beam_h2d = time.perf_counter()

            wrappers.decode_wrapper.plan(
                indptr=indptr_t,
                indices=indices_t,
                last_page_len=lpl_t,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=ps,
                q_data_type=dtype,
                kv_data_type=page_table.store_dtype,
            )
            if trace_on:
                torch.cuda.synchronize()
                _t_per_beam_plan = time.perf_counter()
                self._plan_trace.append({
                    "strategy": "PER_BEAM",
                    "depth": 1,
                    "decomp_ms":         (_t_decomp        - _t_start) * 1000.0,
                    "decomp_hits":       decomp_hits,
                    "decomp_total":      B,
                    "pick_ms":           (_t_pick          - _t_decomp) * 1000.0,
                    "per_beam_build_ms": (_t_per_beam_build- _t_pick) * 1000.0,
                    "per_beam_h2d_ms":   (_t_per_beam_h2d  - _t_per_beam_build) * 1000.0,
                    "per_beam_plan_ms":  (_t_per_beam_plan - _t_per_beam_h2d) * 1000.0,
                    # DEC_TAIL-specific fields zeroed for schema consistency.
                    "prefix_plan_ms": 0.0,
                    "decode_build_ms": 0.0,
                    "decode_plan_ms": 0.0,
                    "prefix_replan": False,
                    "decode_full_plan": False,
                })
            ctx = PagedAttentionContext(
                is_prefill=False,
                page_table=page_table,
                kv_page_indices=write_pi_t,
                kv_page_offsets=write_po_t,
                decode_wrapper=wrappers.decode_wrapper,
            )
            # PER_BEAM uses natural order — no permute needed.
            return StepPlan(ctx=ctx, beam_order_per_prompt=None, pick=pick)

        # DEC_TAIL — prefix + (N-2) intermediate prefills + per-beam decode + merge.
        _DEC_TAIL_STRATEGIES = {
            Strategy.SHARED_2L_DEC_TAIL,
            Strategy.SHARED_3L_DEC_TAIL,
            Strategy.SHARED_4L_DEC_TAIL,
            Strategy.SHARED_5L_DEC_TAIL,
            Strategy.SHARED_6L_DEC_TAIL,
        }
        if pick.strategy in _DEC_TAIL_STRATEGIES:
            # ---- Plan prefix + intermediate prefills via a single
            # FusedMultiLevelCascadeAttentionWrapper(num_levels=depth-1).
            # The cascade wrapper schedules all prefill levels in one
            # kernel launch with the three-pool design that handles
            # mixed CTA_TILE_Q across levels — ~5× faster than the
            # per-level BatchPrefillWithPagedKVCacheWrapper path on
            # long-prefix workloads (see probe_prefix_wrappers.py).
            #
            # The prefix (level 0) layout is determined entirely by
            # per-prompt LCA depth (immutable prefix pages), so when
            # last_lca_per_prompt is unchanged AND depth=2 (no
            # intermediates), the cascade wrapper's plan from the
            # previous step is still valid — skip the re-plan.
            cur_lcas = list(last_lca_per_prompt)
            n_prefill_levels = dispatch_depth - 1  # all levels except per-beam
            need_replan = (
                self._planned_prefix_lca is None
                or self._planned_prefix_lca != cur_lcas
                or dispatch_depth >= 3
            )
            cascade_wrapper = wrappers.dec_tail_cascade_wrappers[dispatch_depth]
            if need_replan:
                qo_arr, kvp_arr, kvi_arr, kvl_arr, _, _ = (
                    _pack_batched_cascade_arrays_any_depth(
                        levels_per_prompt, ps, device,
                        n_levels=dispatch_depth,
                    )
                )
                # The last entry (index dispatch_depth-1) is the per-beam
                # tail level — that goes through the decode kernel, not
                # the cascade. Cascade plans levels [0 .. n_prefill_levels-1].
                cascade_wrapper.plan(
                    qo_indptr_arr=qo_arr[:n_prefill_levels],
                    paged_kv_indptr_arr=kvp_arr[:n_prefill_levels],
                    paged_kv_indices_arr=kvi_arr[:n_prefill_levels],
                    paged_kv_last_page_len=kvl_arr[:n_prefill_levels],
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    page_size=ps,
                    causal=False,
                    q_data_type=dtype,
                    kv_data_type=page_table.store_dtype,
                )
                self._planned_prefix_lca = cur_lcas
            if trace_on:
                torch.cuda.synchronize()
                _t_prefix_plan = time.perf_counter()
                _prefix_replan = need_replan

            # ---- Build decode wrapper plan via numpy (single H2D each). ----
            BK = B * K
            # write_po is uniform per-prompt (off = pos % ps). Build once
            # per prompt and broadcast.
            write_po_np = np.empty(BK, dtype=np.int32)
            write_pi_np = np.empty(BK, dtype=np.int32)
            decode_lpl_np = np.empty(BK, dtype=np.int32)
            decode_indptr_np = np.empty(BK + 1, dtype=np.int32)
            decode_indptr_np[0] = 0

            # First pass: compute per-beam tail-page count to size the
            # indices buffer + fill scalar arrays.
            tail_lens: list[int] = []
            tail_pages_per_beam: list[list[int]] = []
            row = 0
            for b in range(B):
                pos = current_pos[b]
                off = pos % ps
                pli = pos // ps
                order = beam_order_per_prompt[b]
                bp_b = beams_per_prompt[b]
                prefix_len = len(bp_b[0].pages_prefix)
                tail_idx = pli - prefix_len
                _sizes_last, pages_last, lpl_last = (
                    levels_per_prompt[b][-1]
                )
                for cidx in range(K):
                    pages_i = pages_last[cidx]
                    tail_pages_per_beam.append(pages_i)
                    tail_lens.append(len(pages_i))
                    lpl_i = lpl_last[cidx]
                    decode_lpl_np[row] = ps if lpl_i == -1 else lpl_i
                    beam = bp_b[order[cidx]]
                    write_pi_np[row] = beam.pages_tail[tail_idx]
                    write_po_np[row] = off
                    row += 1
            tail_lens_np = np.asarray(tail_lens, dtype=np.int32)
            np.cumsum(tail_lens_np, out=decode_indptr_np[1:])
            total_tail = int(decode_indptr_np[BK])
            decode_indices_np = np.empty(total_tail, dtype=np.int32)
            for i, pages_i in enumerate(tail_pages_per_beam):
                start = decode_indptr_np[i]
                end = decode_indptr_np[i + 1]
                decode_indices_np[start:end] = pages_i

            # Async H2D — torch.from_numpy is zero-copy on CPU side, .to()
            # with non_blocking lets the small copies overlap.
            indptr_t = torch.from_numpy(decode_indptr_np).to(
                device, non_blocking=True,
            )
            indices_t = torch.from_numpy(decode_indices_np).to(
                device, non_blocking=True,
            )
            lpl_t = torch.from_numpy(decode_lpl_np).to(
                device, non_blocking=True,
            )
            write_pi_t = torch.from_numpy(write_pi_np).to(
                device, non_blocking=True,
            )
            write_po_t = torch.from_numpy(write_po_np).to(
                device, non_blocking=True,
            )
            if trace_on:
                torch.cuda.synchronize()
                _t_decode_build = time.perf_counter()

            # Estimate current max per-beam kv_len cheaply (off + (last
            # page index) * page_size). Using current_pos[0] as a proxy
            # since all prompts in this bench have identical pos.
            cur_max_kv = max(current_pos)
            need_decode_full_plan = (
                self._decode_plan_info_cached is None
                or cur_max_kv >= 2 * self._decode_max_kv_at_plan
            )
            if need_decode_full_plan:
                wrappers.decode_wrapper.plan(
                    indptr=indptr_t,
                    indices=indices_t,
                    last_page_len=lpl_t,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    page_size=ps,
                    q_data_type=dtype,
                    kv_data_type=page_table.store_dtype,
                )
                self._decode_plan_info_cached = (
                    wrappers.decode_wrapper._plan_info
                )
                self._decode_max_kv_at_plan = cur_max_kv
            else:
                # Fast path: reuse cached plan_info, swap KV buffers.
                wrappers.decode_wrapper._paged_kv_indptr_buf = indptr_t
                wrappers.decode_wrapper._paged_kv_indices_buf = indices_t
                wrappers.decode_wrapper._paged_kv_last_page_len_buf = lpl_t
                wrappers.decode_wrapper._plan_info = (
                    self._decode_plan_info_cached
                )
            if trace_on:
                torch.cuda.synchronize()
                _t_decode_plan = time.perf_counter()
            ctx = DecodeTailCascadeContext(
                page_table=page_table,
                write_pi=write_pi_t,
                write_po=write_po_t,
                prefix_wrapper=cascade_wrapper,
                decode_wrapper=wrappers.decode_wrapper,
                inter_wrappers=[],
            )
            if trace_on:
                self._plan_trace.append({
                    "strategy": pick.strategy.name,
                    "depth": pick.depth,
                    "decomp_ms": (_t_decomp - _t_start) * 1000.0,
                    "decomp_hits": decomp_hits,
                    "decomp_total": B,
                    "pick_ms": (_t_pick - _t_decomp) * 1000.0,
                    "prefix_plan_ms": (_t_prefix_plan - _t_pick) * 1000.0,
                    "decode_build_ms": (_t_decode_build - _t_prefix_plan) * 1000.0,
                    "decode_plan_ms": (_t_decode_plan - _t_decode_build) * 1000.0,
                    "prefix_replan": _prefix_replan,
                    "decode_full_plan": need_decode_full_plan,
                })
            return StepPlan(
                ctx=ctx,
                beam_order_per_prompt=beam_order_per_prompt,
                pick=pick,
            )

        # SHARED — fused or non-fused cascade depending on pool_count.
        #
        # Two-tier cache:
        #
        # 1. Level-0 pack cache (qo[0], kvp[0], kvi[0], kvl[0] GPU tensors
        #    + their host-side cumsum lists) keyed by (lca_tuple,
        #    dispatch_depth). Level-0 content is determined entirely by
        #    the per-prompt LCA prefix pages (immutable while LCA is
        #    unchanged), and at long L_p the level-0 build dominates
        #    pack time (~15k int32 entries in kvi_arr[0] at K=16/B=8/
        #    L_p=30000).
        #
        # 2. Plan-state cache (handled below at wrapper.plan dispatch).
        #    Skips wrapper.plan() when the per-level qo_indptr_h +
        #    kv_indptr_h are byte-identical to the wrapper's last plan
        #    — schedule is a pure function of those.
        lca_tuple = tuple(last_lca_per_prompt)

        l0_hit = (
            self._shared_l0_cache_key is not None
            and self._shared_l0_cache_key == (lca_tuple, dispatch_depth)
        )
        if l0_hit:
            qo0_t, kvp0_t, kvi0_t, kvl0_t = self._shared_l0_cache_tensors
            qo_arr, kvp_arr, kvi_arr, kvl_arr, qo_arr_h, kvp_arr_h = (
                _pack_batched_cascade_arrays_any_depth(
                    levels_per_prompt, ps, device,
                    n_levels=dispatch_depth,
                    skip_levels=(0,),
                )
            )
            qo_arr[0] = qo0_t
            kvp_arr[0] = kvp0_t
            kvi_arr[0] = kvi0_t
            kvl_arr[0] = kvl0_t
            # level-0 cumsums were skipped (skip_levels=(0,)); fill from cache.
            qo_arr_h[0] = self._shared_l0_cache_h[0]
            kvp_arr_h[0] = self._shared_l0_cache_h[1]
        else:
            qo_arr, kvp_arr, kvi_arr, kvl_arr, qo_arr_h, kvp_arr_h = (
                _pack_batched_cascade_arrays_any_depth(
                    levels_per_prompt, ps, device,
                    n_levels=dispatch_depth,
                )
            )
            self._shared_l0_cache_key = (lca_tuple, dispatch_depth)
            self._shared_l0_cache_tensors = (
                qo_arr[0], kvp_arr[0], kvi_arr[0], kvl_arr[0],
            )
            self._shared_l0_cache_h = (qo_arr_h[0], kvp_arr_h[0])
        if trace_on:
            torch.cuda.synchronize()
            _t_shared_pack = time.perf_counter()

        # Plan-state cache: schedule is a function of (depth, pool_count,
        # t_large, per-level qo_indptr_h, per-level kv_indptr_h). When all
        # match the wrapper's last plan, skip .plan() and just rebuild the
        # 4 input buffers on the wrapper via torch.cat (the wrapper's
        # ``run`` reads ``self._qo_indptr_buf`` etc. and is otherwise
        # state-driven from pool_*_bufs which the cached schedule fixed).
        # Hit rate target: ~85-93% in steady state (misses on off==0 page
        # appends and fork-driven sub-grouping changes).
        plan_key = (dispatch_depth, pick.pool_count, pick.t_large)
        plan_sig = (
            tuple(tuple(q) for q in qo_arr_h),
            tuple(tuple(k) for k in kvp_arr_h),
        )
        plan_hit = False
        if pick.pool_count == 2:
            # MLCA (non-fused) wrapper not yet plan-state-cached. Full
            # re-plan every step.
            active_wrapper = wrappers.cascade_dual_wrappers[dispatch_depth]
            active_wrapper.plan(
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
                kv_data_type=page_table.store_dtype,
            )
        else:
            active_wrapper = wrappers.cascade_wrappers[dispatch_depth]
            cur = self._shared_plan_cache.get(id(active_wrapper))
            plan_hit = cur is not None and cur == (plan_key, plan_sig)
            if plan_hit:
                # Reuse cached schedule. Overwrite only the 4 input buffers;
                # pool_*_bufs / o_indptr_buf / partial_o / partial_lse /
                # num_chunks_per_level on the wrapper are reusable as-is.
                active_wrapper._qo_indptr_buf = torch.cat(qo_arr, dim=0)
                active_wrapper._paged_kv_indptr_buf = torch.cat(kvp_arr, dim=0)
                active_wrapper._paged_kv_indices_buf = torch.cat(kvi_arr, dim=0)
                active_wrapper._paged_kv_last_page_len_buf = torch.cat(
                    kvl_arr, dim=0)
            else:
                # pool=1 forces a single CTA tile size (the picker's chosen
                # T_large); pool=2 (handled in the if-branch above) routes
                # through the dual non-fused path.
                force_t = pick.t_large if pick.pool_count == 1 else None
                active_wrapper.plan(
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
                    kv_data_type=page_table.store_dtype,
                    force_cta_tile_q=force_t,
                )
                self._shared_plan_cache[id(active_wrapper)] = (
                    plan_key, plan_sig,
                )
        if trace_on:
            torch.cuda.synchronize()
            _t_shared_plan = time.perf_counter()

        # write_pi/write_po follow cascade order: for prompt b, the i-th
        # cascade row corresponds to beam at natural index
        # beam_order_per_prompt[b][i]. Identity for d=2.
        write_pi_list = []
        write_po_list = []
        for b in range(B):
            pos = current_pos[b]
            off = pos % ps
            pli = pos // ps
            order = beam_order_per_prompt[b]
            bp_b = beams_per_prompt[b]
            prefix_len = len(bp_b[0].pages_prefix)
            tail_idx = pli - prefix_len
            for cidx in range(K):
                beam = bp_b[order[cidx]]
                write_pi_list.append(beam.pages_tail[tail_idx])
                write_po_list.append(off)
        ctx = AdaptivePoolContext(
            page_table=page_table,
            write_pi=torch.tensor(
                write_pi_list, dtype=torch.int32, device=device),
            write_po=torch.tensor(
                write_po_list, dtype=torch.int32, device=device),
            wrapper=active_wrapper,
        )
        if trace_on:
            torch.cuda.synchronize()
            _t_shared_write = time.perf_counter()
            self._plan_trace.append({
                "strategy": pick.strategy.name,
                "depth": pick.depth,
                "decomp_ms":       (_t_decomp       - _t_start)     * 1000.0,
                "decomp_hits":     decomp_hits,
                "decomp_total":    B,
                "pick_ms":         (_t_pick         - _t_decomp)    * 1000.0,
                "shared_pack_ms":  (_t_shared_pack  - _t_pick)      * 1000.0,
                "shared_plan_ms":  (_t_shared_plan  - _t_shared_pack) * 1000.0,
                "shared_write_ms": (_t_shared_write - _t_shared_plan) * 1000.0,
                "shared_l0_hit":   bool(l0_hit),
                "shared_plan_hit": bool(plan_hit),
            })
        return StepPlan(
            ctx=ctx,
            beam_order_per_prompt=beam_order_per_prompt,
            pick=pick,
        )


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
    # Default to 3: matches the deepest strategy in ``DEFAULT_STRATEGIES``
    # (``SHARED_3L_1POOL``) and the picker's ``max_dispatch_depth=3``
    # cap. Going deeper costs ``_adaptive_levels`` an extra intermediate-
    # split iteration and ``_workload_from_levels`` an extra level
    # entry, neither of which the picker can use. Override if widening
    # ``available_strategies`` to include SHARED_4L+.
    max_cascade_levels: int = 3,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
    return_timings: bool = False,
    return_phase_timings: bool = False,
    coefficients: Coefficients | None = None,
    fused_merge: bool = False,
    available_strategies: set[Strategy] | None = None,
    return_picks: bool = False,
    select_at_prefill: PrefillSelect = standard_prefill_select,
    select_at_decode: DecodeSelect = standard_decode_select,
):
    """bs_kernel beam search — picker chooses (strategy, depth, pool_count,
    t_large) per step from cost_model.pick_strategy_batch.

    Signature matches paged.py / mlca.py / adaptive_pool.py so the existing
    benchmark harness drops in.

    Extra kwargs:
      * ``coefficients`` — calibrated cost-model weights, defaults to the
        per-device cache loaded by ``load_or_defaults(device)``.
      * ``fused_merge`` — cost-model toggle for the merge_us term in the
        SHARED-strategy candidates (used by ablation C3).
      * ``available_strategies`` — restrict the candidate set (e.g. force
        always-SHARED to isolate paged-vs-cascade crossover).
      * ``return_picks`` — return the per-step picks alongside beams.
    """
    coeff = (
        coefficients
        if coefficients is not None
        else load_or_defaults(device)
    )
    backend = BsKernelBackend(
        coefficients=coeff,
        fused_merge=fused_merge,
        available_strategies=available_strategies,
    )
    result = _shared_beam_search(
        model, config, prompt_ids, max_new_tokens, beam_width,
        backend=backend,
        page_size=page_size,
        max_num_pages=max_num_pages,
        max_cascade_levels=max_cascade_levels,
        device=device,
        dtype=dtype,
        kv_dtype=kv_dtype,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        return_picks=return_picks,
        select_at_prefill=select_at_prefill,
        select_at_decode=select_at_decode,
    )
    if int(os.environ.get("BS_KERNEL_TRACE_PLAN", "0")):
        _dump_plan_trace(backend._plan_trace)
    return result


def _dump_plan_trace(trace: list) -> None:
    """Print sub-phase summary of bs_kernel plan_decode_step.

    Splits the trace into DEC_TAIL-path and PER_BEAM-path subsets and
    prints each separately (PER_BEAM has different sub-phases from
    DEC_TAIL).
    """
    import statistics
    if not trace:
        print("[bs_kernel plan-trace] no entries")
        return

    def _fmt_rows(subset, fields, n_total):
        out = []
        n = len(subset)
        out.append(
            f"  {'phase':<22} {'mean':>10} {'median':>10} "
            f"{'p90':>10} {'p99':>10} {'total':>12}"
        )
        for f in fields:
            xs = [r[f] for r in subset]
            xs_sorted = sorted(xs)
            mean = sum(xs) / n
            med = statistics.median(xs)
            p90 = xs_sorted[int(0.9 * (n - 1))]
            p99 = xs_sorted[int(0.99 * (n - 1))]
            total = sum(xs)
            out.append(
                f"  {f:<22} {mean:>10.3f} {med:>10.3f} "
                f"{p90:>10.3f} {p99:>10.3f} {total:>12.2f}"
            )
        return out

    def _is_dec_tail(r):
        return "DEC_TAIL" in r.get("strategy", "")

    pb = [r for r in trace if r.get("strategy") == "PER_BEAM"]
    dt = [r for r in trace if _is_dec_tail(r)]
    sh = [r for r in trace if r.get("strategy", "").startswith("SHARED_")
          and not _is_dec_tail(r)]

    print(f"\n[bs_kernel plan-trace] {len(trace)} steps "
          f"(PER_BEAM: {len(pb)}, DEC_TAIL: {len(dt)}, SHARED: {len(sh)})")

    if dt:
        n = len(dt)
        n_prefix = sum(1 for r in dt if r["prefix_replan"])
        n_decode = sum(1 for r in dt if r["decode_full_plan"])
        print(f"\n  -- DEC_TAIL path ({n} steps) --")
        print(
            f"  prefix re-plans: {n_prefix}/{n} ({100.0*n_prefix/n:.1f}%)   "
            f"decode full re-plans: {n_decode}/{n} ({100.0*n_decode/n:.1f}%)"
        )
        for line in _fmt_rows(dt, (
            "decomp_ms", "pick_ms", "prefix_plan_ms",
            "decode_build_ms", "decode_plan_ms",
        ), n):
            print(line)

    if pb:
        n = len(pb)
        print(f"\n  -- PER_BEAM path ({n} steps) --")
        for line in _fmt_rows(pb, (
            "decomp_ms", "pick_ms",
            "per_beam_build_ms", "per_beam_h2d_ms", "per_beam_plan_ms",
        ), n):
            print(line)

    if sh:
        n = len(sh)
        from collections import Counter
        strat_counts = Counter(r["strategy"] for r in sh)
        l0_hits = sum(1 for r in sh if r.get("shared_l0_hit"))
        plan_hits = sum(1 for r in sh if r.get("shared_plan_hit"))
        print(f"\n  -- SHARED path ({n} steps) --")
        print(f"  strategy mix: {dict(strat_counts)}")
        print(
            f"  level-0 pack cache hits: {l0_hits}/{n} "
            f"({100.0*l0_hits/n:.1f}%)   "
            f"plan_info cache hits: {plan_hits}/{n} "
            f"({100.0*plan_hits/n:.1f}%)"
        )
        for line in _fmt_rows(sh, (
            "decomp_ms", "pick_ms",
            "shared_pack_ms", "shared_plan_ms", "shared_write_ms",
        ), n):
            print(line)

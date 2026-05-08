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


def _collapse_d3_to_d2(levels: list) -> list:
    """Collapse a depth=3 cascade layout to depth=2 in O(K), without redoing
    the LCA scan.

    `_adaptive_levels(max_levels=3)` may emit:
      [L0_shared, L1_intermediate, L2_per_beam]   — len 3
      [L0_shared, L2_per_beam]                    — len 2 (no intermediate)
      [L1_intermediate, L2_per_beam]              — len 2 (no shared)
      [L2_per_beam]                               — len 1

    Only the len==3 case is collapsed: each group's intermediate pages are
    prepended to its member beams' tails, producing
    [L0_shared, L2_with_intermediate_prepended]. Beam order from the d=3
    layout (group order) is preserved by callers — the dispatch already
    permutes query rows via beam_order_per_prompt.

    Cost-model equivalence: ``_per_prompt_levels(w, depth=2)`` folds
    intermediate pages back into each beam's suffix length, which is
    exactly what this function does at the page-list level.
    """
    if len(levels) != 3:
        return levels
    L0_shared, L1_inter, L2_per_beam = levels
    sizes_per_group, inter_pages_per_group, _ = L1_inter
    l2_sizes, per_beam_tail, per_beam_lpl = L2_per_beam
    merged_tail: list[list[int]] = []
    idx = 0
    for gi, gsize in enumerate(sizes_per_group):
        inter_pages = inter_pages_per_group[gi]
        for _ in range(gsize):
            merged_tail.append(inter_pages + per_beam_tail[idx])
            idx += 1
    return [L0_shared, (l2_sizes, merged_tail, per_beam_lpl)]


def _workload_from_levels(
    levels: list,
    K: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> WorkloadShape:
    """Derive the cost model's WorkloadShape from `_adaptive_levels`'s output."""
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
    if len(levels) == 3:
        sizes_mid, pages_mid, _ = levels[1]
        groups = [
            (sizes_mid[gi], len(pages_mid[gi]) * page_size)
            for gi in range(len(sizes_mid))
        ]
        intermediate = IntermediateShape(groups=groups)

    bytes_per_kv = 2 * num_kv_heads * head_dim * dtype_bytes
    return WorkloadShape(
        K=K,
        L_p=L_p,
        suffix_lens=suffix_lens,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_kv=bytes_per_kv,
        intermediate=intermediate,
    )


def _pack_batched_cascade_arrays_any_depth(
    levels_per_prompt: list[list],
    page_size: int,
    device: torch.device,
    *,
    n_levels: int,
):
    """Generalized batched-cascade-array packer for ``n_levels ∈ {2, 3}``.

    Every prompt is expected to contribute exactly ``n_levels`` levels; the
    picker only emits depth=3 when every prompt has an intermediate
    layout, so this invariant holds at the call site.
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
            for s in sizes:
                qo_arr_per_level[li].append(qo_arr_per_level[li][-1] + s)
            for pages, lpl in zip(group_pages, group_lpl):
                kvi_arr_per_level[li].extend(pages)
                kvp_arr_per_level[li].append(
                    kvp_arr_per_level[li][-1] + len(pages))
                kvl_arr_per_level[li].append(
                    page_size if lpl == -1 else lpl)

    qo_arr = [
        torch.tensor(qo_arr_per_level[i], dtype=torch.int32, device=device)
        for i in range(n_levels)
    ]
    kvp_arr = [
        torch.tensor(kvp_arr_per_level[i], dtype=torch.int32, device=device)
        for i in range(n_levels)
    ]
    kvi_arr = [
        torch.tensor(kvi_arr_per_level[i], dtype=torch.int32, device=device)
        for i in range(n_levels)
    ]
    kvl_arr = [
        torch.tensor(kvl_arr_per_level[i], dtype=torch.int32, device=device)
        for i in range(n_levels)
    ]
    return qo_arr, kvp_arr, kvi_arr, kvl_arr


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class _BsKernelWrappers(WrapperBundle):
    """Wrapper bundle for bs_kernel. Wrappers cached at startup; the
    picker chooses one per step.

    For DEC_TAIL strategies the prefix and (optional) intermediate
    levels go through ``prefix_prefill`` / ``inter_prefill`` and the
    tail goes through ``decode_wrapper`` (shared with PER_BEAM).
    """
    __slots__ = (
        "decode_wrapper",
        "cascade_2l",
        "cascade_3l",
        "cascade_dual_2l",
        "cascade_dual_3l",
        "prefix_prefill",
        "inter_prefill",
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
        # Two fused wrappers (depth=2 and depth=3) so we don't reconstruct
        # per step when the picker flips depth.
        wb.cascade_2l = FusedMultiLevelCascadeAttentionWrapper(
            num_levels=2,
            float_workspace_buffer=workspace_buffer,
            kv_layout="NHD",
            device=device,
            max_levels=max_cascade_levels,
        )
        wb.cascade_3l = FusedMultiLevelCascadeAttentionWrapper(
            num_levels=3,
            float_workspace_buffer=workspace_buffer,
            kv_layout="NHD",
            device=device,
            max_levels=max_cascade_levels,
        )
        # Non-fused MLCA wrappers for pool_count=2 picks.
        wb.cascade_dual_2l = MultiLevelCascadeAttentionWrapper(
            num_levels=2,
            float_workspace_buffer=workspace_buffer,
            kv_layout="NHD",
        )
        wb.cascade_dual_3l = MultiLevelCascadeAttentionWrapper(
            num_levels=3,
            float_workspace_buffer=workspace_buffer,
            kv_layout="NHD",
        )
        # Single-level prefill wrappers for the front half of DEC_TAIL.
        # Re-planned per step against (prefix-only) or (intermediate-only)
        # KV layouts.
        wb.prefix_prefill = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout="NHD",
        )
        wb.inter_prefill = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout="NHD",
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
        dtype_bytes = torch.tensor([], dtype=dtype).element_size()

        # ---- Per-prompt: compute up-to-3-level decomposition + workload. ----
        levels_d3_per_prompt: list[list] = []
        beam_order_d3_per_prompt: list[list[int]] = []
        workloads: list[WorkloadShape] = []
        for b in range(B):
            pos = current_pos[b]
            off = pos % ps
            bp_b = beams_per_prompt[b]
            pages_prefix = bp_b[0].pages_prefix
            pages_tails = [bm.pages_tail for bm in bp_b]
            lpl_per_beam = [off + 1] * K
            levels, beam_order, lca_b = _adaptive_levels(
                pages_prefix, pages_tails, lpl_per_beam, K,
                max_levels=3,
                start_lca=last_lca_per_prompt[b],
            )
            last_lca_per_prompt[b] = lca_b
            levels_d3_per_prompt.append(levels)
            beam_order_d3_per_prompt.append(beam_order)
            w = _workload_from_levels(
                levels, K, ps, num_kv_heads, head_dim, dtype_bytes,
            )
            workloads.append(w)

        # ---- Cost-model pick across the whole batch. ----
        pick = pick_strategy_batch(
            workloads, self.coefficients,
            fused_merge=self.fused_merge,
            available_strategies=self.available_strategies,
        )

        # ---- Choose dispatch depth + collapse layouts if needed. ----
        dispatch_depth = pick.depth if pick.share else 2
        if dispatch_depth == 3:
            levels_per_prompt = levels_d3_per_prompt
            beam_order_per_prompt = beam_order_d3_per_prompt
        else:
            levels_per_prompt = [
                _collapse_d3_to_d2(levels_d3_per_prompt[b])
                for b in range(B)
            ]
            beam_order_per_prompt = beam_order_d3_per_prompt

        # ---- Dispatch. ----
        if pick.strategy == Strategy.PER_BEAM:
            # B*K independent paged-decode sequences in one launch.
            flat_indptr = [0]
            flat_indices: list[int] = []
            flat_lpl: list[int] = []
            write_pi_list: list[int] = []
            write_po_list: list[int] = []
            for b in range(B):
                pos = current_pos[b]
                off = pos % ps
                pli = pos // ps
                bp_b = beams_per_prompt[b]
                prefix_len = len(bp_b[0].pages_prefix)
                tail_idx = pli - prefix_len
                for beam in bp_b:
                    flat_indices.extend(beam.pages_prefix)
                    flat_indices.extend(beam.pages_tail)
                    flat_indptr.append(len(flat_indices))
                    flat_lpl.append(off + 1)
                    write_pi_list.append(beam.pages_tail[tail_idx])
                    write_po_list.append(off)
            wrappers.decode_wrapper.plan(
                indptr=torch.tensor(
                    flat_indptr, dtype=torch.int32, device=device),
                indices=torch.tensor(
                    flat_indices, dtype=torch.int32, device=device),
                last_page_len=torch.tensor(
                    flat_lpl, dtype=torch.int32, device=device),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=ps,
            )
            ctx = PagedAttentionContext(
                is_prefill=False,
                page_table=page_table,
                kv_page_indices=torch.tensor(
                    write_pi_list, dtype=torch.int32, device=device),
                kv_page_offsets=torch.tensor(
                    write_po_list, dtype=torch.int32, device=device),
                decode_wrapper=wrappers.decode_wrapper,
            )
            # PER_BEAM uses natural order — no permute needed.
            return StepPlan(ctx=ctx, beam_order_per_prompt=None, pick=pick)

        # DEC_TAIL — prefix(/intermediate) prefill + per-beam decode + merge.
        if pick.strategy in (
            Strategy.SHARED_2L_DEC_TAIL, Strategy.SHARED_3L_DEC_TAIL,
        ):
            # ---- Plan prefix prefill (cache across steps when LCA stable). ----
            # The prefix layout is determined entirely by per-prompt LCA
            # depth (the prefix pages themselves are immutable). When
            # last_lca_per_prompt is unchanged from the previous DEC_TAIL
            # step, the prefix wrapper's plan is still valid — skip the
            # re-plan and the qo/kv array build for the prefix level.
            cur_lcas = list(last_lca_per_prompt)
            need_prefix_replan = (
                self._planned_prefix_lca is None
                or self._planned_prefix_lca != cur_lcas
                or dispatch_depth == 3   # 3L plans both prefix + inter together
            )
            if need_prefix_replan:
                qo_arr, kvp_arr, kvi_arr, kvl_arr = (
                    _pack_batched_cascade_arrays_any_depth(
                        levels_per_prompt, ps, device,
                        n_levels=dispatch_depth,
                    )
                )
                wrappers.prefix_prefill.plan(
                    qo_indptr=qo_arr[0],
                    paged_kv_indptr=kvp_arr[0],
                    paged_kv_indices=kvi_arr[0],
                    paged_kv_last_page_len=kvl_arr[0],
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim,
                    page_size=ps,
                    causal=False,
                    q_data_type=dtype,
                    kv_data_type=dtype,
                )
                inter_w = None
                if dispatch_depth == 3:
                    wrappers.inter_prefill.plan(
                        qo_indptr=qo_arr[1],
                        paged_kv_indptr=kvp_arr[1],
                        paged_kv_indices=kvi_arr[1],
                        paged_kv_last_page_len=kvl_arr[1],
                        num_qo_heads=num_qo_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim_qk=head_dim,
                        page_size=ps,
                        causal=False,
                        q_data_type=dtype,
                        kv_data_type=dtype,
                    )
                    inter_w = wrappers.inter_prefill
                self._planned_prefix_lca = cur_lcas
            else:
                # Reuse last step's plan. inter_w is None at depth=2;
                # depth=3 forces replan above so this branch only handles
                # the depth=2 cache hit.
                inter_w = None

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
            ctx = DecodeTailCascadeContext(
                page_table=page_table,
                write_pi=write_pi_t,
                write_po=write_po_t,
                prefix_wrapper=wrappers.prefix_prefill,
                decode_wrapper=wrappers.decode_wrapper,
                inter_wrapper=inter_w,
            )
            return StepPlan(
                ctx=ctx,
                beam_order_per_prompt=beam_order_per_prompt,
                pick=pick,
            )

        # SHARED — fused or non-fused cascade depending on pool_count.
        qo_arr, kvp_arr, kvi_arr, kvl_arr = (
            _pack_batched_cascade_arrays_any_depth(
                levels_per_prompt, ps, device,
                n_levels=dispatch_depth,
            )
        )
        if pick.pool_count == 2:
            active_wrapper = (
                wrappers.cascade_dual_3l if dispatch_depth == 3
                else wrappers.cascade_dual_2l
            )
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
                kv_data_type=dtype,
            )
        else:
            active_wrapper = (
                wrappers.cascade_3l if dispatch_depth == 3
                else wrappers.cascade_2l
            )
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
                kv_data_type=dtype,
                force_cta_tile_q=force_t,
            )

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
    max_cascade_levels: int = 4,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
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
    return _shared_beam_search(
        model, config, prompt_ids, max_new_tokens, beam_width,
        backend=backend,
        page_size=page_size,
        max_num_pages=max_num_pages,
        max_cascade_levels=max_cascade_levels,
        device=device,
        dtype=dtype,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        return_picks=return_picks,
        select_at_prefill=select_at_prefill,
        select_at_decode=select_at_decode,
    )

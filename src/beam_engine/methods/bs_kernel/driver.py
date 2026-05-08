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

import torch
from dataclasses import dataclass
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
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
    """Wrapper bundle for bs_kernel. Five wrappers cached at startup; the
    picker chooses one per step."""
    __slots__ = (
        "decode_wrapper",
        "cascade_2l",
        "cascade_3l",
        "cascade_dual_2l",
        "cascade_dual_3l",
    )


@dataclass
class BsKernelBackend:
    name: str = "bs_kernel"
    use_split_pages: bool = True

    coefficients: Coefficients | None = None
    fused_merge: bool = False
    available_strategies: set[Strategy] | None = None

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
        return_picks=return_picks,
        select_at_prefill=select_at_prefill,
        select_at_decode=select_at_decode,
    )

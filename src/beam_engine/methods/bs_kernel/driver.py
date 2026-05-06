"""Phase 1 driver for the bs_kernel method.

Wires `cost_model.pick_strategy` to the existing FlashInfer wrappers.
No kernel modifications yet — Phase 1 only validates that the cost
model picks the right *strategy* (PER_BEAM vs SHARED with chosen
depth) on top of unmodified kernels. Phase 2 (kernel mods in
`3rdparty/flashinfer`) honors `t_large` / `pool_count` / `fused_merge`;
in Phase 1 those fields of `Pick` are recorded for diagnostics but do
not change kernel behavior.

Beam-search outer structure mirrors `methods/adaptive_pool.py`. Cascade
plan helpers (`_adaptive_levels`, `_build_cascade_plan`,
`AdaptivePoolContext`, `_PrefillCtx`, `_add_ref`, `_remove_ref`,
`Beam`) are imported unchanged so any plan-building bug fix lands in
both methods at once. Per-beam dispatch reuses
`PagedAttentionContext` from the paged baseline.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    FusedMultiLevelCascadeAttentionWrapper,
)

from ...baselines.paged import PagedAttentionContext
from ...decoding import (
    DecodeSelect,
    PrefillSelect,
    standard_decode_select,
    standard_prefill_select,
)
from ...page_table import PageTable
from ..adaptive_pool import (
    AdaptivePoolContext,
    Beam,
    _PrefillCtx,
    _add_ref,
    _adaptive_levels,
    _remove_ref,
)
from .calibrate import load_or_defaults
from .cost_model import (
    Coefficients,
    IntermediateShape,
    Pick,
    Strategy,
    WorkloadShape,
    pick_strategy,
)


# ---------------------------------------------------------------------------
# Combined LCA + workload + level decomposition (single Python pass).
# ---------------------------------------------------------------------------


def _workload_from_levels(
    levels: list,
    K: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> WorkloadShape:
    """Derive the cost model's WorkloadShape from `_adaptive_levels`'s
    output, avoiding a duplicate LCA scan."""
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
        intermediate = IntermediateShape(
            G=len(sizes_mid),
            group_size=sizes_mid[0],
            inter_len_tokens=len(pages_mid[0]) * page_size,
        )

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


def _pack_cascade_arrays(
    levels: list,
    page_size: int,
    device: torch.device,
):
    """Pack `_adaptive_levels` output into the 4 indptr/indices tensor
    arrays the cascade wrapper consumes. Mirrors the tensor-packing
    half of `adaptive_pool._build_cascade_plan` but skips the
    `_adaptive_levels` call (caller passes the precomputed levels).
    """
    qo_arr: list[torch.Tensor] = []
    kvp_arr: list[torch.Tensor] = []
    kvi_arr: list[torch.Tensor] = []
    kvl_arr: list[torch.Tensor] = []

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
            last_page.append(page_size if lpl == -1 else lpl)
        qo_arr.append(torch.tensor(qo_indptr, dtype=torch.int32, device=device))
        kvp_arr.append(torch.tensor(kv_indptr, dtype=torch.int32, device=device))
        kvi_arr.append(torch.tensor(kv_indices, dtype=torch.int32, device=device))
        kvl_arr.append(torch.tensor(last_page, dtype=torch.int32, device=device))
    return qo_arr, kvp_arr, kvi_arr, kvl_arr


# ---------------------------------------------------------------------------
# Beam-search driver
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
    """Phase 1 bs_kernel beam search.

    Per decode step, calls `cost_model.pick_strategy` on the current
    beam state and dispatches to:
      * PER_BEAM     → BatchDecodeWithPagedKVCacheWrapper
      * SHARED_2L_*  → FusedMultiLevelCascadeAttentionWrapper, depth=2
      * SHARED_3L_*  → FusedMultiLevelCascadeAttentionWrapper, depth=3

    Signature matches `paged.py` / `tree.py` / `fasttree.py` /
    `adaptive_pool.py` so the existing benchmark harness drops in.

    Extra kwargs (Phase-1 specific):
      * ``coefficients`` — calibrated `Coefficients`, defaults to
        `Coefficients.defaults()` until calibration runs.
      * ``fused_merge`` — Phase 2 kernel toggle. Set False here; the
        cost model uses it only to drop the merge_us term when comparing
        strategies (so ablation C3 can switch it on without rerunning
        the kernel).
      * ``available_strategies`` — restrict the choice space (ablation
        switch; e.g. force always-SHARED to isolate paged-vs-cascade
        crossover, or force fixed depth for A2).
      * ``return_picks`` — also return the per-step Pick lists for
        oracle-vs-model comparison (B1).
    """
    # When no coefficients are passed, load the per-device tuned values
    # from cache (`~/.cache/beam_engine/coeffs-<gpu>.json`). Falls back
    # to `Coefficients.defaults()` if the cache is missing.
    coeff = coefficients if coefficients is not None else load_or_defaults(device)
    timings = {"prefill_ms": 0.0, "decode_step_ms": []}
    picks_log: list[list[Pick]] = []

    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers

    K = beam_width
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()

    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device,
    )

    all_beams_out: list[list[Beam]] = []

    for prompt in prompt_ids:
        L_p = len(prompt)
        page_table = PageTable(
            layer_num=num_layers,
            page_size=page_size,
            max_num_pages=max_num_pages,
            head_num=num_kv_heads,
            head_dim=head_dim,
            device=torch.device(device),
            store_dtype=dtype,
        )
        ps = page_table.page_size

        prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout="NHD",
        )
        decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout="NHD", use_tensor_cores=True,
        )
        cascade = FusedMultiLevelCascadeAttentionWrapper(
            num_levels=2,
            float_workspace_buffer=workspace_buffer,
            kv_layout="NHD",
            device=torch.device(device),
            max_levels=max_cascade_levels,
        )

        # ----- Prefill -----
        num_pages = (L_p + ps - 1) // ps
        prompt_pages = [page_table.allocate_block() for _ in range(num_pages)]
        kv_pi = torch.tensor(
            [prompt_pages[i // ps] for i in range(L_p)],
            dtype=torch.int32, device=device,
        )
        kv_po = torch.tensor(
            [i % ps for i in range(L_p)], dtype=torch.int32, device=device,
        )
        prefill_wrapper.plan(
            qo_indptr=torch.tensor([0, L_p], dtype=torch.int32, device=device),
            paged_kv_indptr=torch.tensor(
                [0, num_pages], dtype=torch.int32, device=device,
            ),
            paged_kv_indices=torch.tensor(
                prompt_pages, dtype=torch.int32, device=device,
            ),
            paged_kv_last_page_len=torch.tensor(
                [L_p - (num_pages - 1) * ps], dtype=torch.int32, device=device,
            ),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=ps,
            causal=True,
        )
        pre_ctx = _PrefillCtx(
            page_table=page_table,
            kv_page_indices=kv_pi,
            kv_page_offsets=kv_po,
            wrapper=prefill_wrapper,
        )
        input_ids = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
        positions = torch.arange(L_p, device=device).unsqueeze(0)

        if return_timings:
            torch.cuda.synchronize()
            t_pre = time.perf_counter()

        with torch.no_grad():
            hidden = model.forward(
                input_ids=input_ids, positions=positions, ctx=pre_ctx,
            )
            logits = model.compute_logits(hidden[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_1d = log_probs.squeeze(0)
            topk_lp_1d, topk_ids_1d = select_at_prefill(log_probs_1d, K)
            topk_lp = topk_lp_1d.unsqueeze(0)
            topk_ids = topk_ids_1d.unsqueeze(0)

        if return_timings:
            torch.cuda.synchronize()
            timings["prefill_ms"] += (time.perf_counter() - t_pre) * 1000.0

        beams = [
            Beam(
                token_ids=[topk_ids[0, i].item()],
                cum_log_prob=topk_lp[0, i].item(),
                pages=list(prompt_pages),
            )
            for i in range(K)
        ]
        rc: dict[int, int] = {}
        for p in prompt_pages:
            _add_ref(rc, p, K)
        current_pos = L_p
        per_prompt_picks: list[Pick] = []

        # ----- Decode loop -----
        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                if return_timings:
                    torch.cuda.synchronize()
                    t_step = time.perf_counter()

                # 1) CoW: ensure each beam owns its write page.
                pos = current_pos
                pli = pos // ps
                off = pos % ps
                for beam in beams:
                    if off == 0:
                        new_page = page_table.allocate_block()
                        beam.pages.append(new_page)
                        _add_ref(rc, new_page, 1)
                    else:
                        write_page = beam.pages[pli]
                        if rc[write_page] > 1:
                            new_page = page_table.copy_block(write_page, off)
                            _remove_ref(rc, write_page, page_table)
                            beam.pages[pli] = new_page
                            _add_ref(rc, new_page, 1)

                pages_per_beam = [list(b.pages) for b in beams]
                lpl_per_beam = [off + 1] * K  # all beams just wrote slot off

                # 2) Single LCA + intermediate scan; reuse for both
                # cost-model workload extraction and cascade plan packing.
                levels, beam_order_default = _adaptive_levels(
                    pages_per_beam, lpl_per_beam, K, max_levels=3,
                )
                w = _workload_from_levels(
                    levels, K, ps, num_kv_heads, head_dim, dtype_bytes,
                )
                pick = pick_strategy(
                    w, coeff,
                    fused_merge=fused_merge,
                    available_strategies=available_strategies,
                )
                if return_picks:
                    per_prompt_picks.append(pick)

                # 3) Dispatch.
                if pick.strategy == Strategy.PER_BEAM:
                    flat_indptr = [0]
                    flat_indices: list[int] = []
                    flat_lpl: list[int] = []
                    for beam in beams:
                        flat_indices.extend(beam.pages)
                        flat_indptr.append(len(flat_indices))
                        flat_lpl.append(off + 1)
                    decode_wrapper.plan(
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
                    write_pi = torch.tensor(
                        [b.pages[pli] for b in beams],
                        dtype=torch.int32, device=device,
                    )
                    write_po = torch.tensor(
                        [off] * K, dtype=torch.int32, device=device,
                    )
                    ctx = PagedAttentionContext(
                        is_prefill=False,
                        page_table=page_table,
                        kv_page_indices=write_pi,
                        kv_page_offsets=write_po,
                        decode_wrapper=decode_wrapper,
                    )
                    beam_order = list(range(K))
                else:
                    # If cost model picked depth-2 but our scan found
                    # 3-level structure, redo the scan at max_levels=2
                    # (LCA reused; only intermediate detection is skipped).
                    if pick.depth == 2 and len(levels) == 3:
                        levels, beam_order_default = _adaptive_levels(
                            pages_per_beam, lpl_per_beam, K, max_levels=2,
                        )
                    qo_arr, kvp_arr, kvi_arr, kvl_arr = _pack_cascade_arrays(
                        levels, ps, torch.device(device),
                    )
                    beam_order = beam_order_default
                    n_lvls = len(levels)
                    if cascade.num_levels != n_lvls:
                        cascade = FusedMultiLevelCascadeAttentionWrapper(
                            num_levels=n_lvls,
                            float_workspace_buffer=workspace_buffer,
                            kv_layout="NHD",
                            device=torch.device(device),
                            max_levels=max_cascade_levels,
                        )
                    # Mod 2: cost model picks t_large per step.
                    # Always force the wrapper to honor the cost model's
                    # T choice. Even when pick.pool_count==2, leaving
                    # `force_cta_tile_q=None` lets the wrapper's
                    # `large_cta_tile_q = 128 if max_packed > 64 else 64`
                    # heuristic flip to T=128 at K=65, which is
                    # empirically 30-90% slower than T=64 across our
                    # K∈[17,128] long-prefix regime. The 2-pool win
                    # over 1-pool in cases where it actually wins is
                    # <1%; the K-boundary loss when wrapper auto-picks
                    # T is 30-90%. Net win to force.
                    plan_force_t = pick.t_large
                    cascade.plan(
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
                        force_cta_tile_q=plan_force_t,
                    )
                    write_pi = torch.tensor(
                        [beams[i].pages[pli] for i in beam_order],
                        dtype=torch.int32, device=device,
                    )
                    write_po = torch.tensor(
                        [off] * K, dtype=torch.int32, device=device,
                    )
                    ctx = AdaptivePoolContext(
                        page_table=page_table,
                        write_pi=write_pi,
                        write_po=write_po,
                        wrapper=cascade,
                    )

                # 4) Forward + topk in beam_order space.
                beam_input = torch.tensor(
                    [[beams[i].token_ids[-1]] for i in beam_order],
                    dtype=torch.long, device=device,
                )
                beam_positions = torch.full(
                    (K, 1), current_pos, dtype=torch.long, device=device,
                )
                hidden = model.forward(
                    input_ids=beam_input, positions=beam_positions, ctx=ctx,
                )
                logits = model.compute_logits(hidden[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)
                vocab_size = logits.shape[-1]

                cum_probs = torch.tensor(
                    [beams[i].cum_log_prob for i in beam_order],
                    dtype=torch.float32, device=device,
                )
                scores = cum_probs[:, None] + log_probs.float()
                parent_t, token_t, topk_scores = select_at_decode(scores, K)
                parent_in_order = parent_t.tolist()
                new_token_ids = token_t.tolist()
                scores_list = topk_scores.tolist()
                parent_beam_ids = [beam_order[p] for p in parent_in_order]

                # 5) Refcount surgery + new beam list.
                parent_usage = [0] * K
                for pid in parent_beam_ids:
                    parent_usage[pid] += 1
                for old_idx, usage in enumerate(parent_usage):
                    if usage == 0:
                        for p in beams[old_idx].pages:
                            _remove_ref(rc, p, page_table)
                    elif usage > 1:
                        for p in beams[old_idx].pages:
                            _add_ref(rc, p, usage - 1)
                new_beams: list[Beam] = []
                for i in range(K):
                    pid = parent_beam_ids[i]
                    new_beams.append(
                        Beam(
                            token_ids=beams[pid].token_ids + [new_token_ids[i]],
                            cum_log_prob=scores_list[i],
                            pages=list(beams[pid].pages),
                        )
                    )
                beams = new_beams
                current_pos += 1

                if return_timings:
                    torch.cuda.synchronize()
                    timings["decode_step_ms"].append(
                        (time.perf_counter() - t_step) * 1000.0
                    )

        beams.sort(key=lambda b: b.cum_log_prob, reverse=True)
        all_beams_out.append(beams)
        picks_log.append(per_prompt_picks)

    if return_timings and return_picks:
        return all_beams_out, timings, picks_log
    if return_timings:
        return all_beams_out, timings
    if return_picks:
        return all_beams_out, picks_log
    return all_beams_out

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
    _pack_batched_cascade_arrays,
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
    pick_strategy_batch,
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

    # --------------------------------------------------------------
    # Phase 2: cross-prompt batched cascade. B prompts share one
    # PageTable, one cascade wrapper, and one fused kernel launch per
    # decode step. The cost-model picks one (t_large, pool_count, depth)
    # for the whole batch via pick_strategy_batch — see cost_model.py
    # for why the cross-prompt wave-occupancy effect changes the
    # 1-pool/2-pool boundary at large B.
    # --------------------------------------------------------------
    B = len(prompt_ids)
    L_ps = [len(p) for p in prompt_ids]
    ps = page_size
    # Auto-grow page table for B-prompt batched workloads. Per prompt:
    # ceil(L_p/ps) prefix pages + K * (ceil(max_new/ps) + 1) decode +
    # CoW slack. Bump to whatever the user passed if larger.
    auto_pages = (
        sum((L_p + ps - 1) // ps for L_p in L_ps)
        + B * K * ((max_new_tokens + ps - 1) // ps + 2)
    )
    max_pages_eff = max(max_num_pages, auto_pages + 64)
    page_table = PageTable(
        layer_num=num_layers,
        page_size=ps,
        max_num_pages=max_pages_eff,
        head_num=num_kv_heads,
        head_dim=head_dim,
        device=torch.device(device),
        store_dtype=dtype,
    )

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
    cascade_n_levels = 2

    # ----- Batched prefill (B ragged requests in one launch) -----
    prompt_pages_per_b: list[list[int]] = []
    all_kv_pi: list[int] = []
    all_kv_po: list[int] = []
    qo_indptr_list: list[int] = [0]
    paged_kv_indptr_list: list[int] = [0]
    all_paged_kv_indices: list[int] = []
    paged_kv_lpl_list: list[int] = []
    all_token_ids: list[int] = []
    all_positions: list[int] = []

    for b, L_p in enumerate(L_ps):
        n_pages = (L_p + ps - 1) // ps
        pages = [page_table.allocate_block() for _ in range(n_pages)]
        prompt_pages_per_b.append(pages)
        for i in range(L_p):
            all_kv_pi.append(pages[i // ps])
            all_kv_po.append(i % ps)
        qo_indptr_list.append(qo_indptr_list[-1] + L_p)
        all_paged_kv_indices.extend(pages)
        paged_kv_indptr_list.append(paged_kv_indptr_list[-1] + n_pages)
        paged_kv_lpl_list.append(L_p - (n_pages - 1) * ps)
        all_token_ids.extend(prompt_ids[b])
        all_positions.extend(range(L_p))

    prefill_wrapper.plan(
        qo_indptr=torch.tensor(qo_indptr_list, dtype=torch.int32, device=device),
        paged_kv_indptr=torch.tensor(
            paged_kv_indptr_list, dtype=torch.int32, device=device),
        paged_kv_indices=torch.tensor(
            all_paged_kv_indices, dtype=torch.int32, device=device),
        paged_kv_last_page_len=torch.tensor(
            paged_kv_lpl_list, dtype=torch.int32, device=device),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=ps,
        causal=True,
    )
    pre_ctx = _PrefillCtx(
        page_table=page_table,
        kv_page_indices=torch.tensor(all_kv_pi, dtype=torch.int32, device=device),
        kv_page_offsets=torch.tensor(all_kv_po, dtype=torch.int32, device=device),
        wrapper=prefill_wrapper,
    )
    input_ids = torch.tensor(all_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    positions = torch.tensor(all_positions, dtype=torch.long, device=device).unsqueeze(0)

    if return_timings:
        torch.cuda.synchronize()
        t_pre = time.perf_counter()

    with torch.no_grad():
        hidden = model.forward(
            input_ids=input_ids, positions=positions, ctx=pre_ctx,
        )
        last_indices = [qo_indptr_list[b + 1] - 1 for b in range(B)]
        last_hidden = hidden[0, last_indices, :]            # (B, H)
        logits = model.compute_logits(last_hidden)          # (B, V)
        log_probs = F.log_softmax(logits, dim=-1)           # (B, V)

    if return_timings:
        torch.cuda.synchronize()
        timings["prefill_ms"] = (time.perf_counter() - t_pre) * 1000.0

    # Per-prompt top-K beams (each prompt picks K independent beams).
    beams_per_prompt: list[list[Beam]] = []
    rc_per_prompt: list[dict[int, int]] = []
    for b in range(B):
        top_lp, top_ids = select_at_prefill(log_probs[b], K)
        beams = [
            Beam(
                token_ids=[top_ids[i].item()],
                cum_log_prob=top_lp[i].item(),
                pages=list(prompt_pages_per_b[b]),
            )
            for i in range(K)
        ]
        beams_per_prompt.append(beams)
        rc: dict[int, int] = {}
        for p in prompt_pages_per_b[b]:
            _add_ref(rc, p, K)
        rc_per_prompt.append(rc)

    current_pos: list[int] = list(L_ps)
    picks_per_prompt: list[list[Pick]] = [[] for _ in range(B)]

    # ----- Batched decode loop -----
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if return_timings:
                torch.cuda.synchronize()
                t_step = time.perf_counter()

            # 1) CoW per (prompt, beam).
            for b in range(B):
                pos = current_pos[b]
                pli = pos // ps
                off = pos % ps
                for beam in beams_per_prompt[b]:
                    if off == 0:
                        new_page = page_table.allocate_block()
                        beam.pages.append(new_page)
                        _add_ref(rc_per_prompt[b], new_page, 1)
                    else:
                        write_page = beam.pages[pli]
                        if rc_per_prompt[b][write_page] > 1:
                            new_page = page_table.copy_block(write_page, off)
                            _remove_ref(
                                rc_per_prompt[b], write_page, page_table)
                            beam.pages[pli] = new_page
                            _add_ref(rc_per_prompt[b], new_page, 1)

            # 2) Per-prompt 2-level decomposition + workload.
            levels_per_prompt: list[list] = []
            workloads: list[WorkloadShape] = []
            for b in range(B):
                pos = current_pos[b]
                off = pos % ps
                pages_per_beam = [list(bm.pages) for bm in beams_per_prompt[b]]
                lpl_per_beam = [off + 1] * K
                # Phase 2 batched cascade only handles depth=2 across all
                # prompts (the cost model filters depth=3 unless every
                # prompt has an intermediate, which is rare).
                levels, _ = _adaptive_levels(
                    pages_per_beam, lpl_per_beam, K, max_levels=2,
                )
                levels_per_prompt.append(levels)
                w = _workload_from_levels(
                    levels, K, ps, num_kv_heads, head_dim, dtype_bytes,
                )
                workloads.append(w)

            # 3) Cost-model pick across the whole batch.
            pick = pick_strategy_batch(
                workloads, coeff,
                fused_merge=fused_merge,
                available_strategies=available_strategies,
            )
            if return_picks:
                for b in range(B):
                    picks_per_prompt[b].append(pick)

            # 4) Dispatch.
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
                    for beam in beams_per_prompt[b]:
                        flat_indices.extend(beam.pages)
                        flat_indptr.append(len(flat_indices))
                        flat_lpl.append(off + 1)
                        write_pi_list.append(beam.pages[pli])
                        write_po_list.append(off)
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
                ctx = PagedAttentionContext(
                    is_prefill=False,
                    page_table=page_table,
                    kv_page_indices=torch.tensor(
                        write_pi_list, dtype=torch.int32, device=device),
                    kv_page_offsets=torch.tensor(
                        write_po_list, dtype=torch.int32, device=device),
                    decode_wrapper=decode_wrapper,
                )
            else:
                # Batched cascade: B groups at level 0 (one per prompt's
                # shared prefix), B*K singleton groups at level 1.
                qo_arr, kvp_arr, kvi_arr, kvl_arr = _pack_batched_cascade_arrays(
                    levels_per_prompt, ps, torch.device(device),
                )
                if cascade_n_levels != 2:
                    cascade = FusedMultiLevelCascadeAttentionWrapper(
                        num_levels=2,
                        float_workspace_buffer=workspace_buffer,
                        kv_layout="NHD",
                        device=torch.device(device),
                        max_levels=max_cascade_levels,
                    )
                    cascade_n_levels = 2
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
                    force_cta_tile_q=pick.t_large,
                )
                write_pi_list = []
                write_po_list = []
                for b in range(B):
                    pos = current_pos[b]
                    off = pos % ps
                    pli = pos // ps
                    for beam in beams_per_prompt[b]:
                        write_pi_list.append(beam.pages[pli])
                        write_po_list.append(off)
                ctx = AdaptivePoolContext(
                    page_table=page_table,
                    write_pi=torch.tensor(
                        write_pi_list, dtype=torch.int32, device=device),
                    write_po=torch.tensor(
                        write_po_list, dtype=torch.int32, device=device),
                    wrapper=cascade,
                )

            # 5) Forward over B*K queries (prompt-major / beam-minor).
            all_input: list[list[int]] = []
            all_pos: list[list[int]] = []
            for b in range(B):
                for beam in beams_per_prompt[b]:
                    all_input.append([beam.token_ids[-1]])
                    all_pos.append([current_pos[b]])
            beam_input = torch.tensor(all_input, dtype=torch.long, device=device)
            beam_positions = torch.tensor(all_pos, dtype=torch.long, device=device)
            hidden = model.forward(
                input_ids=beam_input, positions=beam_positions, ctx=ctx,
            )
            logits = model.compute_logits(hidden[:, -1, :])           # (B*K, V)
            log_probs = F.log_softmax(logits, dim=-1)
            vocab_size = logits.shape[-1]
            log_probs_bkv = log_probs.view(B, K, vocab_size)

            # 6) Per-prompt top-K + fork resolution.
            new_beams_per_prompt: list[list[Beam]] = []
            for b in range(B):
                cum_probs = torch.tensor(
                    [bm.cum_log_prob for bm in beams_per_prompt[b]],
                    dtype=torch.float32, device=device,
                )
                scores = cum_probs[:, None] + log_probs_bkv[b].float()
                parent_t, token_t, topk_scores = select_at_decode(scores, K)
                parent_ids = parent_t.tolist()
                new_tokens = token_t.tolist()
                scores_list = topk_scores.tolist()

                parent_usage = [0] * K
                for pid in parent_ids:
                    parent_usage[pid] += 1
                for old_idx, usage in enumerate(parent_usage):
                    if usage == 0:
                        for p in beams_per_prompt[b][old_idx].pages:
                            _remove_ref(rc_per_prompt[b], p, page_table)
                    elif usage > 1:
                        for p in beams_per_prompt[b][old_idx].pages:
                            _add_ref(rc_per_prompt[b], p, usage - 1)
                new_beams: list[Beam] = []
                for i in range(K):
                    pid = parent_ids[i]
                    new_beams.append(
                        Beam(
                            token_ids=beams_per_prompt[b][pid].token_ids
                            + [new_tokens[i]],
                            cum_log_prob=scores_list[i],
                            pages=list(beams_per_prompt[b][pid].pages),
                        )
                    )
                new_beams_per_prompt.append(new_beams)
            beams_per_prompt = new_beams_per_prompt
            for b in range(B):
                current_pos[b] += 1

            if return_timings:
                torch.cuda.synchronize()
                timings["decode_step_ms"].append(
                    (time.perf_counter() - t_step) * 1000.0
                )

    for beams in beams_per_prompt:
        beams.sort(key=lambda b: b.cum_log_prob, reverse=True)

    if return_timings and return_picks:
        return beams_per_prompt, timings, picks_per_prompt
    if return_timings:
        return beams_per_prompt, timings
    if return_picks:
        return beams_per_prompt, picks_per_prompt
    return beams_per_prompt

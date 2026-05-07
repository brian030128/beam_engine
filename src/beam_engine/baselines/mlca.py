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
) -> list[tuple[list[int], list[list[int]], list[int]]]:
    """Always-2-level cascade layout.

    Level 0: longest common page-prefix across all K beams (1 group of K).
    Level 1: each beam's unique tail past the LCA (K singleton groups).

    No heuristic, no group-uniformity detection, no intermediate level.
    """
    # LCA depth — longest common page prefix.
    lca = 0
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
    return levels


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
    levels = _two_level_layout(pages_per_beam, last_page_len_per_beam, K)
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

    Multi-prompt batches run sequentially (one PageTable / cascade wrapper
    per prompt), matching the contract of the tree / FastTree baselines so
    timings are directly comparable.
    """
    timings = {"prefill_ms": 0.0, "decode_step_ms": []}
    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers

    K = beam_width
    B = len(prompt_ids)
    L_ps = [len(p) for p in prompt_ids]
    ps = page_size

    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device,
    )

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
    cascade = MultiLevelCascadeAttentionWrapper(
        num_levels=2,
        float_workspace_buffer=workspace_buffer,
        kv_layout="NHD",
    )

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
        last_hidden = hidden[0, last_indices, :]
        logits = model.compute_logits(last_hidden)
        log_probs = F.log_softmax(logits, dim=-1)

    if return_timings:
        torch.cuda.synchronize()
        timings["prefill_ms"] = (time.perf_counter() - t_pre) * 1000.0

    beams_per_prompt: list[list[Beam]] = []
    rc_per_prompt: list[dict[int, int]] = []
    for b in range(B):
        top_lp, top_ids = select_at_prefill(log_probs[b], K)
        beams_per_prompt.append([
            Beam(
                token_ids=[top_ids[i].item()],
                cum_log_prob=top_lp[i].item(),
                pages=list(prompt_pages_per_b[b]),
            )
            for i in range(K)
        ])
        rc: dict[int, int] = {}
        for p in prompt_pages_per_b[b]:
            _add_ref(rc, p, K)
        rc_per_prompt.append(rc)

    current_pos: list[int] = list(L_ps)

    # ----- Batched decode loop (depth=2 cascade across all B prompts) -----
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if return_timings:
                torch.cuda.synchronize()
                t_step = time.perf_counter()

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

            levels_per_prompt: list[list] = []
            for b in range(B):
                pos = current_pos[b]
                off = pos % ps
                pages_per_beam = [list(bm.pages) for bm in beams_per_prompt[b]]
                lpl_per_beam = [off + 1] * K
                levels_per_prompt.append(
                    _two_level_layout(pages_per_beam, lpl_per_beam, K)
                )

            qo_arr, kvp_arr, kvi_arr, kvl_arr = _pack_batched_cascade_arrays(
                levels_per_prompt, ps, torch.device(device),
            )
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
            )

            write_pi_list: list[int] = []
            write_po_list: list[int] = []
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
            logits = model.compute_logits(hidden[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)
            vocab_size = logits.shape[-1]
            log_probs_bkv = log_probs.view(B, K, vocab_size)

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

    if return_timings:
        return beams_per_prompt, timings
    return beams_per_prompt

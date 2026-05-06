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
    all_beams_out: list[list[Beam]] = []

    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device,
    )

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
        cascade = MultiLevelCascadeAttentionWrapper(
            num_levels=2,  # fixed — see module docstring
            float_workspace_buffer=workspace_buffer,
            kv_layout="NHD",
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

        # K beams initially all share the prompt pages.
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

        # ----- Decode loop -----
        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                if return_timings:
                    torch.cuda.synchronize()
                    t_step = time.perf_counter()

                # 1) Copy-on-write so each beam owns its write page.
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

                # 2) Fixed 2-level cascade plan (LCA prefix + per-beam tails).
                pages_per_beam = [list(b.pages) for b in beams]
                lpl_per_beam = [off + 1] * K  # all beams just wrote slot `off`
                qo_arr, kvp_arr, kvi_arr, kvl_arr = _build_cascade_plan(
                    pages_per_beam,
                    lpl_per_beam,
                    ps,
                    K,
                    device=torch.device(device),
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

                # 4) Build the write-position tensors. With the fixed 2-level
                # layout, beams stay in their natural order — no reordering
                # to align with intermediate-group structure.
                write_pi = torch.tensor(
                    [b.pages[pli] for b in beams],
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

                beam_input = torch.tensor(
                    [[b.token_ids[-1]] for b in beams],
                    dtype=torch.long, device=device,
                )
                beam_positions = torch.full(
                    (K, 1), current_pos, dtype=torch.long, device=device,
                )

                hidden = model.forward(
                    input_ids=beam_input,
                    positions=beam_positions,
                    ctx=ctx,
                )
                logits = model.compute_logits(hidden[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)

                cum_probs = torch.tensor(
                    [b.cum_log_prob for b in beams],
                    dtype=torch.float32, device=device,
                )
                scores = cum_probs[:, None] + log_probs.float()
                parent_t, token_t, topk_scores = select_at_decode(scores, K)
                parent_beam_ids = parent_t.tolist()
                new_token_ids = token_t.tolist()
                scores_list = topk_scores.tolist()

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

    if return_timings:
        return all_beams_out, timings
    return all_beams_out

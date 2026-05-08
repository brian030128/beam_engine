"""Shared page-based beam-search driver.

Four kernel methods (paged, mlca, adaptive_pool, bs_kernel) share the same
PageTable storage model and the same beam-search outer loop:

  prefill → top-K → for step in range(max_new_tokens - 1):
                       CoW → method-specific decode plan → forward → fork

Only the per-step *plan* differs between methods (which wrappers to build,
how to lay out cascade levels or paged-decode arrays, which ``AttentionContext``
to construct). Everything else — workspace allocation, batched prefill,
beam/refcount data structures, copy-on-write, top-K + fork resolution — is
identical scaffolding that has been duplicated across the four files.

This module pulls all of that scaffolding into one ``beam_search(..., backend=...)``
function. Each method supplies a ``PageDecodeBackend`` that exposes:

  * ``use_split_pages`` — whether the beam carries pages in split form
    (``pages_prefix`` shared by reference + per-beam ``pages_tail``) or as
    a single flat ``pages`` list. Cascade-aware methods (mlca / adaptive_pool /
    bs_kernel) use the split form because forks then copy only the small
    tail (≤16 entries) instead of the full ~528-int list. ``paged`` doesn't
    benefit from the split because its kernel input is the full per-beam
    page list anyway.
  * ``init_wrappers(...)`` — construct method-specific FlashInfer wrappers
    once at startup and stash them in an opaque ``WrapperBundle`` the
    planner can reach into. Wrappers are reused across all decode steps.
  * ``plan_decode_step(...)`` — given the current beam state, build the
    per-step ``AttentionContext`` (with ``write_pi`` / ``write_po`` populated)
    plus an optional ``beam_order_per_prompt`` permutation (only bs_kernel's
    depth=3 dispatch produces a non-identity permutation today). Optionally
    returns a ``Pick`` object recorded in ``picks_log`` when ``return_picks``.

The driver carries every host-side optimization that originated in
bs_kernel — numpy-array refcount, split-form pages, batched CoW memcpy,
deferred cum_log_prob materialization, copy-on-fork via ``parent_assignee``,
batched standard-topK, LCA caching across steps. Methods that didn't
previously enjoy these (mlca, adaptive_pool, paged) inherit them for free.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from .decoding import (
    DecodeSelect,
    PrefillSelect,
    standard_decode_select,
    standard_decode_select_batched,
    standard_prefill_select,
)
from .methods.adaptive_pool import Beam, _PrefillCtx
from .models.attention import AttentionContext
from .page_table import PageTable


# ---------------------------------------------------------------------------
# Backend protocol + StepPlan
# ---------------------------------------------------------------------------


@dataclass
class StepPlan:
    """Per-decode-step plan returned by ``backend.plan_decode_step``.

    * ``ctx`` — the ``AttentionContext`` to pass to ``model.forward``. The
      backend has already populated its ``write_pi`` / ``write_po`` tensors
      and any wrapper plan() calls.
    * ``beam_order_per_prompt`` — cascade-order permutation, length-K per
      prompt, when the kernel input requires query rows in non-natural
      order (e.g. depth=3 cascades whose intermediate-group structure
      forces a specific row layout). ``None`` means natural order; the
      driver skips the permute / un-permute entirely.
    * ``pick`` — optional method-specific pick record (bs_kernel's
      ``Pick``); appended to ``picks_log`` when the caller passed
      ``return_picks=True``.
    """
    ctx: AttentionContext
    beam_order_per_prompt: Optional[list[list[int]]] = None
    pick: Any = None


class WrapperBundle:
    """Opaque container for the per-method wrappers a backend constructs at
    startup. Each backend is free to attach whatever attributes it likes;
    the shared driver just hands the bundle back into ``plan_decode_step``.
    """
    pass


class PageDecodeBackend(Protocol):
    name: str
    use_split_pages: bool

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
    ) -> WrapperBundle: ...

    def plan_decode_step(
        self,
        *,
        wrappers: WrapperBundle,
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
    ) -> StepPlan: ...


# ---------------------------------------------------------------------------
# Beam-page helpers — abstract over split vs flat representation
# ---------------------------------------------------------------------------


def _beam_pages_seq(beam: Beam, use_split: bool) -> list[int]:
    """Return the beam's full page list (read-only). Cheap when split:
    just concatenates the prefix list (shared by ref) with the tail.
    Cheap when flat: returns ``beam.pages`` directly.
    """
    if use_split:
        return beam.pages_prefix + beam.pages_tail
    return beam.pages


def _beam_page_at(beam: Beam, idx: int, use_split: bool) -> int:
    """Read the page index at depth ``idx`` in the beam's full page list."""
    if use_split:
        pl = len(beam.pages_prefix)
        if idx < pl:
            return beam.pages_prefix[idx]
        return beam.pages_tail[idx - pl]
    return beam.pages[idx]


def _beam_set_page(beam: Beam, idx: int, value: int, use_split: bool) -> None:
    """Mutate the beam's page list at depth ``idx``. Only the tail is
    mutable in split form (the prefix is shared by reference and never
    written by CoW after prefill — every decode step's writes land in the
    tail by construction)."""
    if use_split:
        pl = len(beam.pages_prefix)
        beam.pages_tail[idx - pl] = value
    else:
        beam.pages[idx] = value


def _beam_append_page(beam: Beam, page: int, use_split: bool) -> None:
    """Append a freshly-allocated page when crossing a page boundary
    (off==0). Goes to the tail in split form, the unified list in flat."""
    if use_split:
        beam.pages_tail.append(page)
    else:
        beam.pages.append(page)


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
    backend: PageDecodeBackend,
    page_size: int = 16,
    max_num_pages: int = 2048,
    max_cascade_levels: int = 4,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    return_timings: bool = False,
    return_phase_timings: bool = False,
    return_picks: bool = False,
    select_at_prefill: PrefillSelect = standard_prefill_select,
    select_at_decode: DecodeSelect = standard_decode_select,
):
    """Page-table-backed batched beam search.

    Returns:
      * ``beams_per_prompt`` (always)
      * ``timings`` if ``return_timings`` (``{prefill_ms, decode_step_ms}``)
      * ``picks_per_prompt`` if ``return_picks`` and the backend records
        picks (e.g. bs_kernel's cost-model picks). Each entry is the per-
        step list of picks the backend emitted for that prompt.

    Per-method behavior is parameterized by ``backend`` (a
    ``PageDecodeBackend`` implementation). See module docstring.
    """
    timings = {"prefill_ms": 0.0, "decode_step_ms": []}
    if return_phase_timings:
        for k in ("cow_ms", "plan_ms", "forward_ms", "topk_ms", "fork_ms"):
            timings[k] = []
    picks_per_prompt: list[list[Any]] = []

    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers

    K = beam_width
    B = len(prompt_ids)
    L_ps = [len(p) for p in prompt_ids]
    ps = page_size
    use_split = backend.use_split_pages

    # --------------------------------------------------------------
    # PageTable allocation. Auto-grow ``max_num_pages`` to fit B
    # prompts plus B*K decode beams. Per prompt:
    #     ceil(L_p/ps) prefix pages + K * (ceil(max_new/ps) + 2)
    # decode pages, plus 64 pages of slack for CoW.
    # --------------------------------------------------------------
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

    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device,
    )
    prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD",
    )
    wrappers = backend.init_wrappers(
        workspace_buffer=workspace_buffer,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=ps,
        max_cascade_levels=max_cascade_levels,
        dtype=dtype,
        device=torch.device(device),
    )

    # --------------------------------------------------------------
    # Batched prefill — B ragged requests in one launch.
    # --------------------------------------------------------------
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
    input_ids = torch.tensor(
        all_token_ids, dtype=torch.long, device=device,
    ).unsqueeze(0)
    positions = torch.tensor(
        all_positions, dtype=torch.long, device=device,
    ).unsqueeze(0)

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

    # --------------------------------------------------------------
    # Initial beam state.
    #
    # Per-prompt top-K. Each prompt's K beams initially share the prompt's
    # prefix pages (refcount = K).
    #
    # Refcount: numpy int32 array of size max_pages_eff (free pages stay
    # at 0). Bulk fork updates use advanced indexing instead of dict
    # loops. ~4× faster than the dict-of-int approach in legacy paged.py /
    # mlca.py / adaptive_pool.py.
    #
    # Pages: split form when the backend supports it (cascade methods),
    # flat form otherwise (paged). Split form lets the prefix be shared
    # by reference across all K beams of a prompt — forks copy only the
    # ≤16-entry tail.
    # --------------------------------------------------------------
    beams_per_prompt: list[list[Beam]] = []
    rc_per_prompt: list[np.ndarray] = []
    for b in range(B):
        top_lp, top_ids = select_at_prefill(log_probs[b], K)
        prompt_prefix = prompt_pages_per_b[b]
        if use_split:
            beams = [
                Beam(
                    token_ids=[top_ids[i].item()],
                    cum_log_prob=top_lp[i].item(),
                    pages_prefix=prompt_prefix,  # shared ref across K
                    pages_tail=[],
                )
                for i in range(K)
            ]
        else:
            beams = [
                Beam(
                    token_ids=[top_ids[i].item()],
                    cum_log_prob=top_lp[i].item(),
                    pages=list(prompt_prefix),
                )
                for i in range(K)
            ]
        beams_per_prompt.append(beams)
        rc = np.zeros(max_pages_eff, dtype=np.int32)
        rc[np.fromiter(
            prompt_prefix, dtype=np.int32, count=len(prompt_prefix),
        )] = K
        rc_per_prompt.append(rc)

    # GPU-resident cumulative log-probs, (B, K). Updated in place each
    # step from torch.topk's output and reused as the additive bias on
    # the next step's logits. Avoids a per-step Python list-comp + CPU→GPU
    # transfer (~1–2 ms/step at K=64). Materialized back to Beam objects
    # at end of decode.
    cum_log_probs_bk = torch.tensor(
        [[beams_per_prompt[b][i].cum_log_prob for i in range(K)]
         for b in range(B)],
        dtype=torch.float32, device=device,
    )

    current_pos: list[int] = list(L_ps)
    if return_picks:
        picks_per_prompt = [[] for _ in range(B)]
    # LCA cache — provably monotone non-decreasing across steps. Initial
    # value = full prompt-prefix length (every beam shares it).
    last_lca_per_prompt: list[int] = [
        len(prompt_pages_per_b[b]) for b in range(B)
    ]

    # --------------------------------------------------------------
    # Decode loop.
    # --------------------------------------------------------------
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if return_timings:
                torch.cuda.synchronize()
                t_step = time.perf_counter()
            if return_phase_timings:
                torch.cuda.synchronize()
                t_phase = time.perf_counter()

            # ---- 1) CoW per (prompt, beam). ----
            # Two regimes:
            #  * off==0: every beam crosses a page boundary; allocate a
            #    fresh page per beam in one ``allocate_blocks(K)`` call.
            #  * off>0: most beams reuse their existing page (rc==1);
            #    only forked beams (rc>1) need a CoW. Collect (src, dst)
            #    page pairs across all prompts and issue ONE batched
            #    fancy-index memcpy per layer at the end. The naive
            #    approach (one memcpy per CoW event per layer) becomes
            #    O(num_layers × num_cow_events) launches per step — at
            #    K=64/B=32/forks-heavy that's tens of thousands of small
            #    launches and was the source of the 133 ms p99 spikes.
            cow_src_pages: list[int] = []
            cow_dst_pages: list[int] = []
            cow_length = 0  # all CoW events this step share length == off
            for b in range(B):
                pos = current_pos[b]
                pli = pos // ps
                off = pos % ps
                rc_b = rc_per_prompt[b]
                bp = beams_per_prompt[b]
                if off == 0:
                    new_pages = page_table.allocate_blocks(K)
                    for beam, new_page in zip(bp, new_pages):
                        _beam_append_page(beam, new_page, use_split)
                        rc_b[new_page] = 1
                else:
                    for beam in bp:
                        write_page = _beam_page_at(beam, pli, use_split)
                        if rc_b[write_page] > 1:
                            new_page = page_table.allocate_block()
                            cow_src_pages.append(write_page)
                            cow_dst_pages.append(new_page)
                            cow_length = off
                            rc_b[write_page] -= 1
                            if rc_b[write_page] == 0:
                                page_table.free_block(int(write_page))
                            _beam_set_page(beam, pli, new_page, use_split)
                            rc_b[new_page] = 1

            if cow_src_pages:
                src_t = torch.tensor(
                    cow_src_pages, device=device, dtype=torch.long,
                )
                dst_t = torch.tensor(
                    cow_dst_pages, device=device, dtype=torch.long,
                )
                for layer_idx in range(page_table.layer_num):
                    kv = page_table.kv_cache_at_layer[layer_idx]
                    kv[dst_t, :, :cow_length] = kv[src_t, :, :cow_length]

            if return_phase_timings:
                torch.cuda.synchronize()
                _t = time.perf_counter()
                timings["cow_ms"].append((_t - t_phase) * 1000.0)
                t_phase = _t

            # ---- 2) Method-specific decode plan. ----
            plan = backend.plan_decode_step(
                wrappers=wrappers,
                beams_per_prompt=beams_per_prompt,
                current_pos=current_pos,
                page_table=page_table,
                K=K,
                B=B,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=ps,
                dtype=dtype,
                device=torch.device(device),
                last_lca_per_prompt=last_lca_per_prompt,
            )
            ctx = plan.ctx
            beam_order_per_prompt = plan.beam_order_per_prompt
            if return_picks and plan.pick is not None:
                for b in range(B):
                    picks_per_prompt[b].append(plan.pick)

            needs_permute = beam_order_per_prompt is not None and any(
                order != list(range(K)) for order in beam_order_per_prompt
            )

            # ---- 3) Forward. ----
            # Cascade methods that emit a non-trivial beam_order need
            # query rows in cascade order; PER_BEAM and depth=2 cascades
            # emit identity order.
            all_input: list[list[int]] = []
            all_pos: list[list[int]] = []
            if needs_permute:
                for b in range(B):
                    order = beam_order_per_prompt[b]
                    for cidx in range(K):
                        beam = beams_per_prompt[b][order[cidx]]
                        all_input.append([beam.token_ids[-1]])
                        all_pos.append([current_pos[b]])
            else:
                for b in range(B):
                    for beam in beams_per_prompt[b]:
                        all_input.append([beam.token_ids[-1]])
                        all_pos.append([current_pos[b]])
            beam_input = torch.tensor(all_input, dtype=torch.long, device=device)
            beam_positions = torch.tensor(all_pos, dtype=torch.long, device=device)

            if return_phase_timings:
                torch.cuda.synchronize()
                _t = time.perf_counter()
                timings["plan_ms"].append((_t - t_phase) * 1000.0)
                t_phase = _t

            hidden = model.forward(
                input_ids=beam_input, positions=beam_positions, ctx=ctx,
            )
            logits = model.compute_logits(hidden[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)
            vocab_size = logits.shape[-1]
            log_probs_bkv_dispatched = log_probs.view(B, K, vocab_size)

            if return_phase_timings:
                torch.cuda.synchronize()
                _t = time.perf_counter()
                timings["forward_ms"].append((_t - t_phase) * 1000.0)
                t_phase = _t

            # Un-permute back to natural beam order so per-prompt top-K
            # operates on natural-indexed scores.
            if needs_permute:
                log_probs_bkv = torch.empty_like(log_probs_bkv_dispatched)
                for b in range(B):
                    order = beam_order_per_prompt[b]
                    if order == list(range(K)):
                        log_probs_bkv[b] = log_probs_bkv_dispatched[b]
                    else:
                        inv = [0] * K
                        for cidx, nat in enumerate(order):
                            inv[nat] = cidx
                        perm = torch.tensor(inv, dtype=torch.long, device=device)
                        log_probs_bkv[b] = log_probs_bkv_dispatched[b][perm]
            else:
                log_probs_bkv = log_probs_bkv_dispatched

            # ---- 4) Top-K + fork resolution. ----
            scores_bkv = cum_log_probs_bk[:, :, None] + log_probs_bkv.float()
            if select_at_decode is standard_decode_select:
                # Fast path: one batched topk with a single D2H sync at
                # the end (vs three syncs per prompt for the per-prompt
                # path).
                parents_t, tokens_t, scores_t = standard_decode_select_batched(
                    scores_bkv, K,
                )
                cum_log_probs_bk = scores_t
                packed = torch.stack([parents_t, tokens_t], dim=0).tolist()
                parents_b = packed[0]
                tokens_b = packed[1]
            else:
                # Slow path: any non-standard select_at_decode (e.g. DBS)
                # carries per-prompt state across groups inside one prompt,
                # so we must call it per-prompt.
                parents_b = [None] * B  # type: ignore[list-item]
                tokens_b = [None] * B   # type: ignore[list-item]
                new_cum_rows: list[torch.Tensor] = []
                for b in range(B):
                    parent_t, token_t, topk_scores = select_at_decode(
                        scores_bkv[b], K,
                    )
                    new_cum_rows.append(topk_scores)
                    parents_b[b] = parent_t.tolist()
                    tokens_b[b] = token_t.tolist()
                cum_log_probs_bk = torch.stack(new_cum_rows, dim=0)

            if return_phase_timings:
                torch.cuda.synchronize()
                _t = time.perf_counter()
                timings["topk_ms"].append((_t - t_phase) * 1000.0)
                t_phase = _t

            # ---- 5) Apply forks (refcount surgery + new beam list). ----
            new_beams_per_prompt: list[list[Beam]] = []
            for b in range(B):
                parent_ids = parents_b[b]
                new_tokens = tokens_b[b]
                rc_b = rc_per_prompt[b]
                bp = beams_per_prompt[b]

                # No-fork fast path: parent_ids is a permutation. Skip
                # parent_usage / rc updates entirely.
                no_fork = len(set(parent_ids)) == K
                if no_fork:
                    new_beams = [bp[pid] for pid in parent_ids]
                    for beam, tok in zip(new_beams, new_tokens):
                        beam.token_ids.append(tok)
                    new_beams_per_prompt.append(new_beams)
                    continue

                parent_usage = [0] * K
                for pid in parent_ids:
                    parent_usage[pid] += 1

                # Bulk rc update over forked/dropped beams' page lists.
                # Split form: only the per-beam tail needs rc updates
                # because the prefix is shared by ref across all K beams
                # of the prompt — sum(usage - 1) over old beams equals 0,
                # so the prefix rc is invariant under top-K. Walk only
                # the ≤16-entry tail per affected beam instead of the
                # full ~528-int unified pages list.
                # Flat form: walk the full pages list (no shared-prefix
                # invariant). Same bulk-numpy update either way.
                freed_pages: list[int] = []
                for old_idx, usage in enumerate(parent_usage):
                    if usage == 1:
                        continue
                    delta = usage - 1
                    if use_split:
                        plist = bp[old_idx].pages_tail
                    else:
                        plist = bp[old_idx].pages
                    if not plist:
                        continue
                    pages_np = np.fromiter(
                        plist, dtype=np.int32, count=len(plist),
                    )
                    if delta < 0:
                        rc_b[pages_np] += delta
                        freed_mask = rc_b[pages_np] == 0
                        if freed_mask.any():
                            freed_pages.extend(
                                pages_np[freed_mask].tolist()
                            )
                    else:
                        rc_b[pages_np] += delta
                for p in freed_pages:
                    page_table.free_block(p)

                # Copy-on-fork beam construction with parent_assignee
                # optimization: each parent's last child takes ownership
                # of the parent's mutable lists; earlier siblings copy
                # the parent's still-intact lists. Saves ~K (Beam(...))
                # constructions per step and reuses Python list objects.
                # cum_log_prob is deferred — kept on GPU as
                # cum_log_probs_bk and materialized at end of decode.
                parent_assignee: dict[int, int] = {}
                for i, pid in enumerate(parent_ids):
                    parent_assignee[pid] = i

                new_beams: list[Beam] = [None] * K  # type: ignore[list-item]
                for i in range(K):
                    pid = parent_ids[i]
                    parent = bp[pid]
                    if i == parent_assignee[pid]:
                        parent.token_ids.append(new_tokens[i])
                        new_beams[i] = parent
                    else:
                        new_t = list(parent.token_ids)
                        new_t.append(new_tokens[i])
                        if use_split:
                            new_beams[i] = Beam(
                                token_ids=new_t,
                                cum_log_prob=0.0,
                                pages_prefix=parent.pages_prefix,
                                pages_tail=list(parent.pages_tail),
                            )
                        else:
                            new_beams[i] = Beam(
                                token_ids=new_t,
                                cum_log_prob=0.0,
                                pages=list(parent.pages),
                            )
                new_beams_per_prompt.append(new_beams)
            beams_per_prompt = new_beams_per_prompt
            for b in range(B):
                current_pos[b] += 1

            if return_phase_timings:
                torch.cuda.synchronize()
                _t = time.perf_counter()
                timings["fork_ms"].append((_t - t_phase) * 1000.0)

            if return_timings:
                torch.cuda.synchronize()
                timings["decode_step_ms"].append(
                    (time.perf_counter() - t_step) * 1000.0
                )

    # End-of-decode materialization. The decode loop deferred per-Beam
    # cum_log_prob updates (kept on GPU as cum_log_probs_bk) and, in
    # split-page methods, kept the unified ``pages`` list empty. Restore
    # both so callers reading the Beam dataclass see the expected fields.
    final_cum = cum_log_probs_bk.tolist()
    for b in range(B):
        beams_b = beams_per_prompt[b]
        scores_b_final = final_cum[b]
        for i, beam in enumerate(beams_b):
            beam.cum_log_prob = scores_b_final[i]
            if use_split:
                beam.pages = beam.pages_prefix + beam.pages_tail

    for beams in beams_per_prompt:
        beams.sort(key=lambda b: b.cum_log_prob, reverse=True)

    if return_timings and return_picks:
        return beams_per_prompt, timings, picks_per_prompt
    if return_timings:
        return beams_per_prompt, timings
    if return_picks:
        return beams_per_prompt, picks_per_prompt
    return beams_per_prompt

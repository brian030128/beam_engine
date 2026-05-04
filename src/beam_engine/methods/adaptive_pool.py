"""Our method — adaptive 2-pool routing + persistent merge kernel.

Idea. At every decode step the beams' KV cache decomposes naturally into two
pools:

  * **Shared pool** — the pages that *every* live beam reads. Right after
    prefill this is just the prompt; after the first decode step it's the
    prompt + whatever shared-suffix pages survived forking. We compute it
    each step as the longest-common page prefix (LCA) over the K beams'
    page lists.
  * **Per-beam pool** — each beam's unique tail (the pages past the LCA),
    sized 1–few pages each.

Routing the two pools through one ``FusedMultiLevelCascadeAttentionWrapper``
(2 cascade levels) gives us:

  * **One HBM load** of the shared prefix per step, regardless of K — the
    win paged-attention can't get because it treats each beam as an
    independent sequence.
  * **A persistent / single-launch kernel** — the fused wrapper (in the
    brian030128/flashinfer fork) collapses both levels' per-CTA work and
    the LSE merge into one launch instead of ``2L − 1`` launches the
    standard cascade wrapper does. CTAs route through the in-kernel pool
    of tile sizes (T=16 for tiny per-beam tiles, T=64/128 for the wide
    shared-prefix tile), which is the actual "2-pool routing" the kernel
    performs.

Adaptiveness. ``num_levels`` per call is chosen each step:

  * 2 — default, just shared LCA + per-beam tail.
  * 3 — when the K beams cleanly partition into G groups (each of size
    K/G) that share an intermediate stretch of pages past the LCA (this
    happens after a fork that keeps multiple children of the same parent
    alive). The third level captures that intermediate sharing, which
    further reduces HBM traffic.

We reuse the ``PageTable`` (refcounted CoW, page_size=16) from the paged
baseline so the allocator behavior is identical.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import flashinfer.page
import torch
import torch.nn.functional as F
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    FusedMultiLevelCascadeAttentionWrapper,
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
    wrapper: FusedMultiLevelCascadeAttentionWrapper
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
# Adaptive 2-pool routing — choose cascade level layout from beam page lists
# ---------------------------------------------------------------------------


def _adaptive_levels(
    pages_per_beam: list[list[int]],
    last_page_len_per_beam: list[int],
    K: int,
    *,
    max_levels: int,
) -> list[tuple[list[int], list[list[int]], list[int]]]:
    """Decide cascade level layout for the current step.

    Returns a list of (group_sizes, group_pages, group_lpl), one entry per
    level (top-down: L0 = shared root). Each level partitions the K beams
    into groups, and each group has one page list + one last-page-len.

    Layout rule:
      * Level 0 (shared) — longest common page-prefix across all K beams.
        Its `last_page_len` is the *shared* part of the trailing page;
        we conservatively pass the common page's full size (page_size) when
        all beams agree on it being a "completed" page (no one has appended
        a divergent slot in it yet) — see _compute_shared_lpl.
      * Last level — per-beam unique tail (one group per beam, with that
        beam's true last_page_len).
      * Optional intermediate level — when the beams partition into G
        groups (G > 1, K % G == 0) that all share a non-trivial run of
        pages past the LCA. Detected by hashing each beam's page list at
        depth lca_depth and checking the partition is uniform; we extend
        the intermediate run as far as members of each group continue to
        agree.
    """
    # 1) LCA depth — longest common page prefix.
    lca = 0
    min_len = min(len(p) for p in pages_per_beam)
    while lca < min_len:
        first = pages_per_beam[0][lca]
        if all(p[lca] == first for p in pages_per_beam):
            lca += 1
        else:
            break

    shared_pages = pages_per_beam[0][:lca]
    # Shared lpl: if no beam has appended into the lca-th-from-front page
    # yet (i.e., the last shared page is a fully-populated previous page),
    # use page_size. The shared region only includes whole pages.
    # We always take all `lca` pages as full (they couldn't be shared if
    # any beam diverged on a slot inside them — CoW would have split).
    page_size = -1  # filled by caller; placeholder used in lpl computation

    levels: list[tuple[list[int], list[list[int]], list[int]]] = []

    # 2) Try to find a uniform intermediate group structure (3-level).
    intermediate: tuple[int, list[list[int]], int] | None = None
    if max_levels >= 3 and lca < min_len:
        # Group beams by their page at depth lca.
        first_div: dict[int, list[int]] = defaultdict(list)
        for i, p in enumerate(pages_per_beam):
            first_div[p[lca]].append(i)
        sizes = {len(v) for v in first_div.values()}
        # Real intermediate only if 1 < G < K (every group has ≥2 beams).
        if len(sizes) == 1 and 1 < len(first_div) < K:
            G = len(first_div)
            # Beams in each group must continue to agree on subsequent pages
            # for the intermediate run.
            group_lists = list(first_div.values())
            run_len = 1
            while True:
                d = lca + run_len
                ok = True
                for grp in group_lists:
                    if d >= min(len(pages_per_beam[b]) for b in grp):
                        ok = False
                        break
                    pivot = pages_per_beam[grp[0]][d]
                    if any(pages_per_beam[b][d] != pivot for b in grp):
                        ok = False
                        break
                if not ok:
                    break
                run_len += 1
            # Each group's intermediate pages.
            intermediate_pages = [
                pages_per_beam[grp[0]][lca : lca + run_len] for grp in group_lists
            ]
            intermediate = (G, intermediate_pages, run_len)
            # Beams reordered per group so the per-beam tail level uses the
            # same row ordering — caller is responsible for emitting Q rows
            # in this order. We put the order in the levels metadata via
            # group_sizes; but the grouping in level 1 implies row layout.
            # For simplicity here we don't reorder rows — instead we emit
            # qo_indptr per level honoring the natural beam order, which
            # the *non-fused* cascade wrapper supports. The fused wrapper
            # also supports per-level qo_indptr arrays. So this works.
            # Override `group_lists` ordering via levels metadata below.

    # Build the levels.
    # Level 0: 1 group of K beams sharing `shared_pages`.
    if shared_pages:
        levels.append(([K], [shared_pages], [page_size]))  # lpl=page_size sentinel

    if intermediate is not None:
        G, inter_pages_per_group, run_len = intermediate
        # Group sizes: beams in group g = K // G
        sizes_per_group = [K // G] * G
        # Intermediate pages are also fully-populated past pages (beams
        # continue past them), so lpl = page_size sentinel.
        levels.append((sizes_per_group, inter_pages_per_group, [page_size] * G))
        # Final per-beam tail level — emit beams in the same group order
        # used for level 1 so the row layout lines up.
        first_div: dict[int, list[int]] = defaultdict(list)
        for i, p in enumerate(pages_per_beam):
            first_div[p[lca]].append(i)
        beam_order: list[int] = []
        for grp in first_div.values():
            beam_order.extend(grp)
        per_beam_tail = [
            pages_per_beam[i][lca + run_len:] for i in beam_order
        ]
        per_beam_lpl = [last_page_len_per_beam[i] for i in beam_order]
        levels.append(([1] * K, per_beam_tail, per_beam_lpl))
        return levels, beam_order

    # Final level: per-beam unique tail past LCA.
    per_beam_tail = [pages_per_beam[i][lca:] for i in range(K)]
    per_beam_lpl = list(last_page_len_per_beam)
    levels.append(([1] * K, per_beam_tail, per_beam_lpl))
    beam_order = list(range(K))
    return levels, beam_order


def _build_cascade_plan(
    pages_per_beam: list[list[int]],
    last_page_len_per_beam: list[int],
    page_size: int,
    K: int,
    *,
    max_levels: int,
    device: torch.device,
):
    """Run adaptive routing and pack the per-level (qo_indptr, kv_indptr,
    kv_indices, last_page_len) arrays. Returns the four arrays, the
    re-ordered beam list, and the chosen num_levels.
    """
    levels, beam_order = _adaptive_levels(
        pages_per_beam, last_page_len_per_beam, K, max_levels=max_levels,
    )
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
    return qo_arr, kvp_arr, kvi_arr, kvl_arr, beam_order, len(levels)


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
):
    """Adaptive 2-pool beam search with the fused multi-level cascade kernel.

    Multi-prompt batches run sequentially (one PageTable / cascade wrapper
    per prompt), matching the contract of the tree / FastTree baselines so
    timings are directly comparable.

    ``max_cascade_levels`` is passed to ``FusedMultiLevelCascadeAttentionWrapper``
    as the JIT compile bound. The actual num_levels chosen each step is
    2 (default) or 3 when the beams partition into uniform shared-intermediate
    groups (the only multi-group structure beam search produces — fork
    children inherit the same parent's pages until they themselves diverge).
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
        cascade = FusedMultiLevelCascadeAttentionWrapper(
            num_levels=2,  # bump per-step via re-init when 3-level fires
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
            topk_lp, topk_ids = log_probs.topk(K, dim=-1)

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

                # 2) Adaptive cascade plan.
                pages_per_beam = [list(b.pages) for b in beams]
                lpl_per_beam = [off + 1] * K  # all beams just wrote slot `off`
                qo_arr, kvp_arr, kvi_arr, kvl_arr, beam_order, num_levels = (
                    _build_cascade_plan(
                        pages_per_beam,
                        lpl_per_beam,
                        ps,
                        K,
                        max_levels=max_cascade_levels,
                        device=torch.device(device),
                    )
                )
                # 3) (Re-)create the wrapper if num_levels changed.
                if cascade.num_levels != num_levels:
                    cascade = FusedMultiLevelCascadeAttentionWrapper(
                        num_levels=num_levels,
                        float_workspace_buffer=workspace_buffer,
                        kv_layout="NHD",
                        device=torch.device(device),
                        max_levels=max_cascade_levels,
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

                # 4) Build the write-position tensors in beam_order so the
                # K queries the model emits line up with the K beams the
                # cascade plan expects.
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

                beam_input = torch.tensor(
                    [[beams[i].token_ids[-1]] for i in beam_order],
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
                vocab_size = logits.shape[-1]

                cum_probs = torch.tensor(
                    [beams[i].cum_log_prob for i in beam_order],
                    dtype=torch.float32, device=device,
                )
                scores = cum_probs[:, None] + log_probs.float()
                topk_scores, topk_flat = scores.reshape(-1).topk(K)
                ids = torch.stack(
                    (topk_flat // vocab_size, topk_flat % vocab_size), dim=0,
                ).tolist()
                # parent index in beam_order space → translate to original beam idx.
                parent_in_order = ids[0]
                new_token_ids = ids[1]
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

    if return_timings:
        return all_beams_out, timings
    return all_beams_out

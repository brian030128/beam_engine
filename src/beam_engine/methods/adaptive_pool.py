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

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import flashinfer.page
import torch

# Module-level trace list, gated by BS_KERNEL_TRACE_CASCADE_KERNELS=1.
# Symmetric with decode_tail_context.DEC_TAIL_KERNEL_TRACE for the
# sub-kernel comparison benchmark.
CASCADE_KERNEL_TRACE: list[dict] = []
import torch.nn.functional as F
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    FusedMultiLevelCascadeAttentionWrapper,
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
            paged_kv_cache=(kv_cache[0], kv_cache[1]),
            kv_indices=self.kv_page_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=self.kv_page_offsets,
            kv_layout="NHD",
        )
        q_3d = q.view(-1, num_heads, head_dim)
        out = self.wrapper.run(q_3d, (kv_cache[0], kv_cache[1]))
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

        # CUDA-event tracing for the cascade kernel, gated by env var.
        # Used by sub-kernel profiling benchmarks to compare 1POOL fused
        # cascade against DEC_TAIL's multi-kernel split (see
        # decode_tail_context.DEC_TAIL_KERNEL_TRACE).
        trace = bool(int(os.environ.get("BS_KERNEL_TRACE_CASCADE_KERNELS", "0")))
        if trace:
            ev = lambda: torch.cuda.Event(enable_timing=True)
            e0, e_append, e_run = ev(), ev(), ev()
            e0.record()

        flashinfer.page.append_paged_kv_cache(
            append_key=k_3d,
            append_value=v_3d,
            batch_indices=batch_idx,
            positions=self.write_po,
            paged_kv_cache=(kv_cache[0], kv_cache[1]),
            kv_indices=self.write_pi,
            kv_indptr=kv_indptr,
            kv_last_page_len=self.write_po,
            kv_layout="NHD",
        )
        if trace:
            e_append.record()

        q_3d = q.view(-1, num_heads, head_dim)
        out = self.wrapper.run(q_3d, (kv_cache[0], kv_cache[1]))
        if trace:
            e_run.record()
            torch.cuda.synchronize()
            CASCADE_KERNEL_TRACE.append({
                "layer_idx": int(layer_idx),
                "append_ms": e0.elapsed_time(e_append),
                "cascade_ms": e_append.elapsed_time(e_run),
            })
        return out.reshape(*q.shape[:-1], num_heads * head_dim)


# ---------------------------------------------------------------------------
# Beam state + page-refcount helpers (shared with paged.py's idea)
# ---------------------------------------------------------------------------


@dataclass
class Beam:
    """Per-beam state carried across decode steps.

    Two parallel representations of the beam's KV pages list coexist:

    * ``pages: list[int]`` — the canonical unified list. Used by all
      legacy methods (paged, mlca, tree, fasttree, the legacy
      adaptive_pool decode loop) and by callers that read the final
      output. Mutated in place by those methods' ``page_update``.

    * ``pages_prefix`` + ``pages_tail`` — split form used by the
      bs_kernel driver to avoid the per-fork ~528-int list copy. The
      prefix holds the prompt's prefix pages by *reference*, shared
      across all K beams in a prompt; the tail holds decode-added
      pages (≤ 16 entries by step 256) and is private per beam. Forks
      copy only the tail. The driver materialises ``pages =
      pages_prefix + pages_tail`` once at end of decode for output
      compatibility.

    A method picks one representation; they are not kept in sync
    during decode. Drivers using the split form should leave
    ``pages = []`` and only set the prefix/tail; legacy methods leave
    ``pages_prefix`` / ``pages_tail`` empty.
    """
    token_ids: list[int]
    cum_log_prob: float
    pages: list[int] = field(default_factory=list)
    pages_prefix: list[int] = field(default_factory=list)
    pages_tail: list[int] = field(default_factory=list)


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
    pages_prefix: list[int],
    pages_tails: list[list[int]],
    last_page_len_per_beam: list[int],
    K: int,
    *,
    max_levels: int,
    start_lca: int = 0,
):
    """Decide cascade level layout for the current step.

    Takes the per-beam page list in *split* form: ``pages_prefix`` is
    the (typically prompt-derived) shared prefix, identical by
    reference across all K beams; ``pages_tails[i]`` holds beam i's
    private decode-added pages. The legacy unified-list form is
    available via :func:`_adaptive_levels_unified`.

    Returns ``(levels, beam_order, lca)`` where ``levels`` is a list of
    ``(group_sizes, group_pages, group_lpl)`` (one entry per cascade
    level, top-down with L0 = shared root), ``beam_order`` is a length-K
    permutation aligning per-beam tails to their position within the
    cascade dispatch, and ``lca`` is the discovered shared-prefix depth
    in pages.

    Layout rule (generalized for ``max_levels >= 2``):
      * Level 0 (shared) — longest common page-prefix across all K beams.
      * Levels 1..N-2 (intermediate) — successive sub-group decompositions
        detected by recursive first-divergent-page splits within each
        group. ``max_levels`` caps the depth; we stop earlier if no
        group at the current depth can split further.
      * Level N-1 (per-beam tail) — one group per beam.

    Singletons (groups that reach size 1 mid-decomposition) emit
    placeholder entries at subsequent intermediate levels so the row
    layout stays consistent across all levels. Their pages-so-far list
    is propagated unchanged at deeper levels.

    The ``start_lca`` parameter lets callers seed the LCA scan from a
    cached value (LCA is monotone non-decreasing across decode steps).
    """
    pl = len(pages_prefix)
    # 1) LCA depth — start from max(start_lca, pl).
    lca = max(start_lca, pl)
    # K=1: skip the LCA extension. With a single beam the `all(...)`
    # quantifier is trivially True, so the loop would walk to the end
    # of the tail and leave the per-beam-tail level with 0 pages but
    # ``last_page_len = off + 1 > 0``. FlashInfer's BatchPrefill plan
    # then computes ``kv_len = (num_pages-1)*page_size + lpl`` < 0 and
    # the cascade launch faults with an illegal memory access. (See
    # exp1a multi_chain_reasoning stage-2 mlca crash.) bs_kernel/
    # adaptive_pool sidestep this via PER_BEAM / fused-wrapper fallbacks
    # but the unfused MLCA path doesn't.
    if K > 1:
        min_tail_len = min(len(t) for t in pages_tails)
        min_len = pl + min_tail_len
        while lca < min_len:
            idx = lca - pl
            first = pages_tails[0][idx]
            if all(t[idx] == first for t in pages_tails):
                lca += 1
            else:
                break

    # shared_pages: prefix + the (typically empty) common tail prefix.
    if lca == pl:
        shared_pages = pages_prefix
    else:
        shared_pages = pages_prefix + pages_tails[0][:lca - pl]
    page_size = -1  # placeholder; filled by caller

    levels: list[tuple[list[int], list[list[int]], list[int]]] = []

    def _page_at(beam_i: int, d: int) -> int:
        if d < pl:
            return pages_prefix[d]
        return pages_tails[beam_i][d - pl]

    # Tail length in pages for each beam (clamps depth probes).
    beam_tail_end = [pl + len(pages_tails[i]) for i in range(K)]

    # Build the levels.
    # Level 0: 1 group of K beams sharing `shared_pages`.
    if shared_pages:
        levels.append(([K], [shared_pages], [page_size]))

    # Iterative sub-grouping for levels 1..max_levels-2. Each "group state"
    # tracks the beams it contains and the page offset where the per-beam
    # divergence resumes. depth=2 → no intermediate levels emitted; depth=N
    # → up to N-2 intermediate levels emitted (fewer if no further splits
    # are detectable).
    @dataclass
    class _GroupState:
        beams: list[int]    # beams in this group, in original index order
        start: int          # absolute page index where this group's tail begins

    current: list[_GroupState] = [
        _GroupState(beams=list(range(K)), start=lca),
    ]

    n_intermediate_target = max(0, max_levels - 2)
    for depth in range(n_intermediate_target):
        if not current:
            break
        # Detect first-divergent partitions per group; compute run_len.
        # If no group splits, stop emitting intermediate levels.
        next_groups: list[_GroupState] = []
        new_sizes: list[int] = []
        new_pages: list[list[int]] = []
        new_lpls: list[int] = []
        any_split = False
        for grp in current:
            if len(grp.beams) <= 1 or grp.start >= min(
                beam_tail_end[b] for b in grp.beams
            ):
                # Singleton or out-of-pages: keep as-is, emit a 0-page
                # placeholder at this level. This preserves the
                # row layout across levels.
                next_groups.append(grp)
                new_sizes.append(len(grp.beams))
                new_pages.append([])  # 0-page contribution at this level
                new_lpls.append(page_size)
                continue

            # Partition by first-divergent page at grp.start.
            first_div: dict[int, list[int]] = defaultdict(list)
            for b in grp.beams:
                first_div[_page_at(b, grp.start)].append(b)
            sub_groups = sorted(first_div.values(), key=lambda g: g[0])

            if len(sub_groups) == 1:
                # All agree at grp.start → run extends; not a real split
                # but we still need to emit something at this level. Treat
                # as one sub-group, find its run_len. (The benefit of
                # extending is captured by sharing a longer page run.)
                sg = sub_groups[0]
                run_len = 1
                while True:
                    d = grp.start + run_len
                    if d >= min(beam_tail_end[b] for b in sg):
                        break
                    pivot = _page_at(sg[0], d)
                    if any(_page_at(b, d) != pivot for b in sg):
                        break
                    run_len += 1
                sub_pages = pages_tails[sg[0]][grp.start - pl : grp.start - pl + run_len]
                new_sizes.append(len(sg))
                new_pages.append(sub_pages)
                new_lpls.append(page_size)
                next_groups.append(_GroupState(
                    beams=sg, start=grp.start + run_len,
                ))
                continue

            # True split: multiple sub-groups.
            any_split = True
            for sg in sub_groups:
                if len(sg) == 1:
                    run_len = 1
                else:
                    run_len = 1
                    while True:
                        d = grp.start + run_len
                        if d >= min(beam_tail_end[b] for b in sg):
                            break
                        pivot = _page_at(sg[0], d)
                        if any(_page_at(b, d) != pivot for b in sg):
                            break
                        run_len += 1
                sub_pages = pages_tails[sg[0]][grp.start - pl : grp.start - pl + run_len]
                new_sizes.append(len(sg))
                new_pages.append(sub_pages)
                new_lpls.append(page_size)
                next_groups.append(_GroupState(
                    beams=sg, start=grp.start + run_len,
                ))

        if not any_split:
            # No group split at this depth — adding the level would only
            # serialize what was already a single per-group LCA extension.
            # Stop emitting intermediates here; current `current` becomes
            # the input to the per-beam-tail level.
            break

        # Sanity: only commit this level if it provides bandwidth savings.
        # (We require at least one sub-group to be non-singleton AND the
        # total group count to exceed the previous level's group count.)
        any_viable_sub = any(s >= 2 for s in new_sizes)
        if not any_viable_sub:
            # All sub-groups are singletons — equivalent to going straight
            # to per-beam tail. Don't emit this level.
            break

        levels.append((new_sizes, new_pages, new_lpls))
        current = next_groups

    # Final level: per-beam unique tails past each group's `start`.
    beam_order: list[int] = []
    per_beam_tail: list[list[int]] = []
    per_beam_lpl: list[int] = []
    for grp in current:
        for b in grp.beams:
            tail_start_in_tail = grp.start - pl
            beam_order.append(b)
            per_beam_tail.append(pages_tails[b][tail_start_in_tail:])
            per_beam_lpl.append(last_page_len_per_beam[b])
    levels.append(([1] * K, per_beam_tail, per_beam_lpl))
    return levels, beam_order, lca


def _adaptive_levels_unified(
    pages_per_beam: list[list[int]],
    last_page_len_per_beam: list[int],
    K: int,
    *,
    max_levels: int,
    start_lca: int = 0,
):
    """Backwards-compat wrapper for callers using a unified pages list.

    Treats every beam's full page list as its tail, with an empty
    prefix. Behavior is identical to the pre-split-form code path.
    Used by the legacy ``adaptive_pool`` decode loop and by
    ``_build_cascade_plan`` (which serves the prefill cascade plan,
    where there's no notion of a shared decode-prefix yet).
    """
    return _adaptive_levels(
        pages_prefix=[],
        pages_tails=pages_per_beam,
        last_page_len_per_beam=last_page_len_per_beam,
        K=K,
        max_levels=max_levels,
        start_lca=start_lca,
    )


def _pack_batched_cascade_arrays(
    levels_per_prompt: list[list],
    page_size: int,
    device: torch.device,
):
    """Concatenate per-prompt 2-level cascade decompositions into one batched
    cascade plan. Each prompt contributes 1 group at level 0 (its shared
    prefix) and K singleton groups at level 1 (per-beam tails). qo_indptr
    enumerates queries in prompt-major / beam-minor order.

    Only depth=2 is supported here. depth=3 batching would require all
    prompts to agree on the same intermediate layout, which is rare for
    deterministic beam search.
    """
    qo_arr_per_level: list[list[int]] = [[0], [0]]
    kvp_arr_per_level: list[list[int]] = [[0], [0]]
    kvi_arr_per_level: list[list[int]] = [[], []]
    kvl_arr_per_level: list[list[int]] = [[], []]

    for levels in levels_per_prompt:
        if len(levels) == 1:
            sizes_last, group_pages_last, group_lpl_last = levels[0]
            sizes_shared, group_pages_shared, group_lpl_shared = (
                [0], [[]], [-1],
            )
        else:
            sizes_shared, group_pages_shared, group_lpl_shared = levels[0]
            sizes_last, group_pages_last, group_lpl_last = levels[-1]

        for s in sizes_shared:
            qo_arr_per_level[0].append(qo_arr_per_level[0][-1] + s)
        for pages, lpl in zip(group_pages_shared, group_lpl_shared):
            kvi_arr_per_level[0].extend(pages)
            kvp_arr_per_level[0].append(
                kvp_arr_per_level[0][-1] + len(pages))
            kvl_arr_per_level[0].append(
                page_size if lpl == -1 else lpl)
        for s in sizes_last:
            qo_arr_per_level[1].append(qo_arr_per_level[1][-1] + s)
        for pages, lpl in zip(group_pages_last, group_lpl_last):
            kvi_arr_per_level[1].extend(pages)
            kvp_arr_per_level[1].append(
                kvp_arr_per_level[1][-1] + len(pages))
            kvl_arr_per_level[1].append(
                page_size if lpl == -1 else lpl)

    qo_arr = [
        torch.tensor(qo_arr_per_level[i], dtype=torch.int32, device=device)
        for i in range(2)
    ]
    kvp_arr = [
        torch.tensor(kvp_arr_per_level[i], dtype=torch.int32, device=device)
        for i in range(2)
    ]
    kvi_arr = [
        torch.tensor(kvi_arr_per_level[i], dtype=torch.int32, device=device)
        for i in range(2)
    ]
    kvl_arr = [
        torch.tensor(kvl_arr_per_level[i], dtype=torch.int32, device=device)
        for i in range(2)
    ]
    return qo_arr, kvp_arr, kvi_arr, kvl_arr


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
    levels, beam_order, _lca = _adaptive_levels_unified(
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
# Backend
# ---------------------------------------------------------------------------


from dataclasses import dataclass as _dataclass


class _AdaptivePoolWrappers:
    """Wrapper bundle: one fused 2-level cascade wrapper, reused across steps."""
    __slots__ = ("cascade",)


@_dataclass
class AdaptivePoolBackend:
    name: str = "adaptive_pool"
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
        wb = _AdaptivePoolWrappers()
        wb.cascade = FusedMultiLevelCascadeAttentionWrapper(
            num_levels=2,
            float_workspace_buffer=workspace_buffer,
            kv_layout="NHD",
            device=device,
            max_levels=max_cascade_levels,
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
            pages_prefix = bp_b[0].pages_prefix
            pages_tails = [bm.pages_tail for bm in bp_b]
            lpl_per_beam = [off + 1] * K
            # Force depth=2 in batched mode — depth=3 batching requires
            # all prompts to agree on intermediate layout, rare for
            # deterministic decoding.
            levels, _beam_order, lca_b = _adaptive_levels(
                pages_prefix, pages_tails, lpl_per_beam, K,
                max_levels=2,
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
    max_cascade_levels: int = 4,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    return_timings: bool = False,
    select_at_prefill: PrefillSelect = standard_prefill_select,
    select_at_decode: DecodeSelect = standard_decode_select,
):
    """Adaptive 2-pool beam search with the fused multi-level cascade kernel.

    Always 2 cascade levels in batched mode — depth=3 would require all B
    prompts to agree on intermediate layout, which is rare for deterministic
    decoding. (Single-prompt depth=3 dispatch lives in bs_kernel via the
    cost-model picker.)
    """
    from ..page_driver import beam_search as _shared_beam_search

    return _shared_beam_search(
        model, config, prompt_ids, max_new_tokens, beam_width,
        backend=AdaptivePoolBackend(),
        page_size=page_size,
        max_num_pages=max_num_pages,
        max_cascade_levels=max_cascade_levels,
        device=device,
        dtype=dtype,
        return_timings=return_timings,
        select_at_prefill=select_at_prefill,
        select_at_decode=select_at_decode,
    )

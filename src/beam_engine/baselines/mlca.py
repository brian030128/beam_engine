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
import numpy as np
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

        k_3d = k.reshape(-1, num_kv_heads, head_dim)
        v_3d = v.reshape(-1, num_kv_heads, head_dim)
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
        q_3d = q.reshape(-1, num_heads, head_dim)
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
    wrapper: MultiLevelCascadeAttentionWrapper
    _write_helper_indptr: torch.Tensor | None = field(default=None, repr=False)

    def attend(self, q, k, v, layer_idx):
        num_kv_heads = self.page_table.head_num
        head_dim = self.page_table.head_dim
        num_heads = q.shape[-1] // head_dim
        kv_cache = self.page_table.kv_cache_at_layer[layer_idx]

        k_3d = k.reshape(-1, num_kv_heads, head_dim)
        v_3d = v.reshape(-1, num_kv_heads, head_dim)
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
            paged_kv_cache=(kv_cache[0], kv_cache[1]),
            kv_indices=self.write_pi,
            kv_indptr=kv_indptr,
            kv_last_page_len=self.write_po,
            kv_layout="NHD",
        )

        q_3d = q.reshape(-1, num_heads, head_dim)
        out = self.wrapper.run(q_3d, (kv_cache[0], kv_cache[1]))
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
    *,
    start_lca: int = 0,
):
    """Always-2-level cascade layout.

    Level 0: longest common page-prefix across all K beams (1 group of K).
    Level 1: each beam's unique tail past the LCA (K singleton groups).

    No heuristic, no group-uniformity detection, no intermediate level.

    Returns ``(levels, lca)``. The optional ``start_lca`` lets callers
    seed the LCA scan from a value cached at the previous decode step
    (LCA is provably monotone non-decreasing across steps).
    """
    # LCA depth — longest common page prefix. Resume from the cached
    # value when the caller has one.
    lca = start_lca
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
    return levels, lca


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
    levels, _lca = _two_level_layout(pages_per_beam, last_page_len_per_beam, K)
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
# Backend
# ---------------------------------------------------------------------------


from dataclasses import dataclass as _dataclass


# ---------------------------------------------------------------------------
# SM90 plan-state cache helpers
# ---------------------------------------------------------------------------


def _is_sm90(device: torch.device) -> bool:
    """True if the device is a Hopper (SM90) GPU.

    The non-fused ``BatchPrefillWithPagedKVCacheWrapper`` dispatches to
    ``PrefillSM90Plan`` on SM90+, which is the path whose schedule bakes
    in token-level kv_len. On other architectures the planner is
    cumsum-shape-invariant and the kv_len patch is unnecessary (and the
    SM90Info field layout doesn't apply).
    """
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor) == (9, 0)


# Index of kv_len_offset inside PrefillPlanSM90Info::ToVector() output, see
# 3rdparty/flashinfer/include/flashinfer/attention/scheduler.cuh struct
# definition. Indices [0..8] = [qo_tile_indices_offset, qo_indptr_offset,
# kv_indptr_offset, qo_len_offset, kv_len_offset, head_indices_offset,
# work_indptr_offset, batch_indices_offset, same_schedule_for_all_heads].
_SM90_QO_INDPTR_OFFSET_IDX = 1
_SM90_KV_INDPTR_OFFSET_IDX = 2
_SM90_KV_LEN_OFFSET_IDX = 4
_SM90_WORK_INDPTR_OFFSET_IDX = 6
_SM90_BATCH_INDICES_OFFSET_IDX = 7


def _build_sm90_patch_state(sub_wrapper, num_sm90_ctas: int):
    """Read the C++ scheduler's batch_indices mapping out of the
    sub-wrapper's pinned-host int workspace, and pre-bake a device-side
    int32 view that maps each work item back to its request index.

    The schedule is in ``_pin_memory_int_workspace_buffer`` (host-pinned
    mirror of ``_int_workspace_buffer``; both populated by the C++ plan
    via ``cudaMemcpyAsync``). batch_indices[work_idx] = request_idx,
    where total_works = work_indptr[num_sm90_ctas].

    Returns ``(kv_len_offset, total_works, batch_indices_dev)`` or
    ``None`` if the sub-wrapper isn't on the SM90 path (no plan_info or
    different layout).
    """
    plan_info = getattr(sub_wrapper, "_plan_info", None)
    if plan_info is None or len(plan_info) != 9:
        return None
    pin = sub_wrapper._pin_memory_int_workspace_buffer
    int_ws = sub_wrapper._int_workspace_buffer
    if pin is None or int_ws is None:
        return None
    qo_indptr_offset = int(plan_info[_SM90_QO_INDPTR_OFFSET_IDX])
    kv_indptr_offset = int(plan_info[_SM90_KV_INDPTR_OFFSET_IDX])
    kv_len_offset = int(plan_info[_SM90_KV_LEN_OFFSET_IDX])
    work_indptr_offset = int(plan_info[_SM90_WORK_INDPTR_OFFSET_IDX])
    batch_indices_offset = int(plan_info[_SM90_BATCH_INDICES_OFFSET_IDX])
    # work_indptr has num_sm90_ctas + 1 int32 entries. Last entry is
    # total_works. Read it off the pinned host buffer to avoid a
    # device→host sync.
    pin_i32 = pin.view(torch.int32)
    work_end = (work_indptr_offset // 4) + num_sm90_ctas + 1
    total_works = int(pin_i32[work_end - 1].item())
    batch_indices_h = pin_i32[
        batch_indices_offset // 4 : batch_indices_offset // 4 + total_works
    ].clone()
    batch_indices_dev = batch_indices_h.to(int_ws.device, non_blocking=True)
    return (
        qo_indptr_offset,
        kv_indptr_offset,
        kv_len_offset,
        total_works,
        batch_indices_dev,
    )


def _apply_sm90_schedule_patch(
    sub_wrapper,
    patch_state,
    new_qo_indptr_per_request: torch.Tensor,
    new_kv_indptr_per_request: torch.Tensor,
    new_kv_lens_per_request: torch.Tensor,
) -> None:
    """In-place update the per-work qo_indptr, kv_indptr, and kv_len
    arrays baked into the SM90 schedule.

    All three are stored as int32 vectors of length ``total_works`` in
    ``_int_workspace_buffer`` at the respective offsets. The kernel
    consumes them per CTA to know which token range to read for the
    request that CTA serves. Under uniform +1 kv_len ticks per request,
    the SM90 scheduler's work assignment (which CTA serves which
    request) is invariant; only these three indexed-by-work tensors
    change. So patching them is sufficient to skip the C++ scheduler.

    Args (all int32 on device, length = batch_size for the level):
      - new_qo_indptr_per_request[i]  = qo_indptr_h[i] (start of request i in qo)
      - new_kv_indptr_per_request[i]  = kv_indptr_h[i] (start of request i in kv pages)
      - new_kv_lens_per_request[i]    = token-level kv_len for request i

    See PrefillSM90Plan in
    3rdparty/flashinfer/include/flashinfer/attention/scheduler.cuh
    lines 940-944 for the per-cta_idx push of these three fields.
    """
    (qo_indptr_offset, kv_indptr_offset, kv_len_offset,
     total_works, batch_indices_dev) = patch_state
    int_ws = sub_wrapper._int_workspace_buffer
    view = int_ws.view(torch.int32)
    # Gather per-work values from per-request arrays.
    qo_per_work = new_qo_indptr_per_request[batch_indices_dev]
    kvp_per_work = new_kv_indptr_per_request[batch_indices_dev]
    kvl_per_work = new_kv_lens_per_request[batch_indices_dev]
    qo_base = qo_indptr_offset // 4
    kvp_base = kv_indptr_offset // 4
    kvl_base = kv_len_offset // 4
    view[qo_base : qo_base + total_works].copy_(qo_per_work, non_blocking=True)
    view[kvp_base : kvp_base + total_works].copy_(kvp_per_work, non_blocking=True)
    view[kvl_base : kvl_base + total_works].copy_(kvl_per_work, non_blocking=True)


def _compute_kv_lens_from_paged(
    paged_kv_indptr: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Token-level KV length per request from page indptr + last_page_len.

    Mirrors ``flashinfer.page.get_seq_lens``: clamp(page_count - 1, 0) *
    page_size + last_page_len, returned as int32.
    """
    page_counts = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
    return (
        torch.clamp(page_counts - 1, min=0) * page_size
        + paged_kv_last_page_len
    ).to(torch.int32)


class _MlcaWrappers:
    __slots__ = ("cascade",)


@_dataclass
class MlcaBackend:
    name: str = "mlca"
    use_split_pages: bool = True
    # Set ``MLCA_PLAN_CACHE=1`` (and we're on SM90) to enable the SM90
    # plan-state cache. See ``_apply_sm90_kv_len_patch`` for the trick:
    # the SM90 prefill scheduler bakes per-CTA-work token-level kv_len
    # into ``_int_workspace_buffer``; on a cumsum-matching cache hit we
    # only need to patch those baked kv_len values. Everything else
    # (qo/kv indptr per CTA, batch_indices, work_indptr) is invariant
    # under uniform +1 kv-length shifts within a page epoch.
    enable_plan_cache: bool = field(
        default_factory=lambda: bool(int(
            __import__("os").environ.get("MLCA_PLAN_CACHE", "0")
        )),
    )
    _plan_trace: list = field(default_factory=list, init=False, repr=False)
    # plan_sig (per-level qo/kvp cumsum tuples) → per-sub-wrapper patch
    # state. None means cache disabled or first call (miss).
    _plan_cache: dict = field(default_factory=dict, init=False, repr=False)
    _plan_hits: int = field(default=0, init=False, repr=False)
    _plan_misses: int = field(default=0, init=False, repr=False)

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
        wb = _MlcaWrappers()
        # Non-fused cascade — runs (2L-1) launches per step (L attn + L-1
        # merges) where L=2: 3 launches per step total.
        wb.cascade = MultiLevelCascadeAttentionWrapper(
            num_levels=2,
            float_workspace_buffer=workspace_buffer,
            kv_layout="NHD",
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

        import os as _os
        trace_on = bool(int(_os.environ.get("MLCA_TRACE_PLAN", "0")))
        if trace_on:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        ps = page_size

        # ----- Numpy-bridge build of the packed 2-level cascade arrays. -----
        # Replaces the per-prompt _adaptive_levels + _pack_batched_cascade_arrays
        # pipeline. We exploit the fixed 2-level MLCA layout:
        #   Level 0 (shared): K beams per prompt; kv = pages_prefix (full
        #     pages — the page_driver fix moves the prompt's partial page
        #     into each beam's pages_tail, so pages_prefix here is all-full).
        #   Level 1 (per-beam tail): 1 beam per group; kv = pages_tail.
        # The _adaptive_levels LCA-extension that folds naturally-shared tail
        # pages into level 0 is dropped — beams' tails are essentially always
        # divergent on the workloads MLCA is benched on (the SGLang two-stage
        # harness pre-allocates a distinct tail per leaf; CoW keeps them
        # distinct afterward). last_lca_per_prompt is no longer maintained
        # here; the driver's initial value is ignored by this path.
        BK = B * K
        qo0 = np.arange(0, BK + 1, K, dtype=np.int32)
        qo1 = np.arange(0, BK + 1, dtype=np.int32)

        shared_lens = np.empty(B, dtype=np.int32)
        tail_lens = np.empty(BK, dtype=np.int32)
        for b in range(B):
            bp_b = beams_per_prompt[b]
            shared_lens[b] = len(bp_b[0].pages_prefix)
            base = b * K
            for k in range(K):
                tail_lens[base + k] = len(bp_b[k].pages_tail)

        kvp0 = np.empty(B + 1, dtype=np.int32)
        kvp0[0] = 0
        np.cumsum(shared_lens, out=kvp0[1:])
        kvp1 = np.empty(BK + 1, dtype=np.int32)
        kvp1[0] = 0
        np.cumsum(tail_lens, out=kvp1[1:])

        kvi0 = np.empty(int(kvp0[-1]), dtype=np.int32)
        pos_w = 0
        for b in range(B):
            pp = beams_per_prompt[b][0].pages_prefix
            n = len(pp)
            if n:
                kvi0[pos_w:pos_w + n] = pp
                pos_w += n

        kvi1 = np.empty(int(kvp1[-1]), dtype=np.int32)
        pos_w = 0
        for b in range(B):
            bp_b = beams_per_prompt[b]
            for k in range(K):
                pt = bp_b[k].pages_tail
                n = len(pt)
                if n:
                    kvi1[pos_w:pos_w + n] = pt
                    pos_w += n

        # Last-page-len: level 0 is all-full → page_size; level 1 is the
        # post-write last_page_len = (pos % ps) + 1.
        kvl0 = np.full(B, ps, dtype=np.int32)
        kvl1 = np.empty(BK, dtype=np.int32)
        for b in range(B):
            kvl1[b * K:(b + 1) * K] = (current_pos[b] % ps) + 1

        if trace_on:
            _t_numpy = time.perf_counter()

        qo_arr = [
            torch.from_numpy(qo0).to(device, non_blocking=True),
            torch.from_numpy(qo1).to(device, non_blocking=True),
        ]
        kvp_arr = [
            torch.from_numpy(kvp0).to(device, non_blocking=True),
            torch.from_numpy(kvp1).to(device, non_blocking=True),
        ]
        kvi_arr = [
            torch.from_numpy(kvi0).to(device, non_blocking=True),
            torch.from_numpy(kvi1).to(device, non_blocking=True),
        ]
        kvl_arr = [
            torch.from_numpy(kvl0).to(device, non_blocking=True),
            torch.from_numpy(kvl1).to(device, non_blocking=True),
        ]
        if trace_on:
            torch.cuda.synchronize()
            _t_h2d = time.perf_counter()

        # SM90 plan-state cache: under uniform per-request kv_len ticks
        # (every decode step adds 1 token to every active request), the
        # SM90 scheduler's work distribution is invariant. So once we've
        # planned this wrappers.cascade for this (B, K) configuration,
        # we just need to patch the per-work qo_indptr, kv_indptr, and
        # kv_len arrays — never replan. The signature is just
        # (B, num_l1_requests = B*K) to detect a configuration change.
        plan_sig = None
        plan_hit = False
        if self.enable_plan_cache and _is_sm90(device):
            plan_sig = (B, BK)
            cached = self._plan_cache.get(id(wrappers.cascade))
            if cached is not None and cached[0] == plan_sig:
                # Cache hit: patch kv_len in each sub-wrapper's
                # int_workspace_buffer, then overwrite the 4 input
                # buffers so wrapper.run reads fresh data.
                patches = cached[1]
                # Compute per-request kv_len (token-level) for each
                # level from the new cumsum + last_page_len arrays
                # (already on device as kvp_arr[i] / kvl_arr[i]).
                for i, sub in enumerate(
                    wrappers.cascade._batch_prefill_wrappers
                ):
                    # Per-request starts (drop the final cumsum entry to
                    # match the C++ scheduler's per-batch-i indexing).
                    qo_per_req = qo_arr[i][:-1]
                    kvp_per_req = kvp_arr[i][:-1]
                    kv_lens = _compute_kv_lens_from_paged(
                        kvp_arr[i], kvl_arr[i], ps,
                    )
                    _apply_sm90_schedule_patch(
                        sub, patches[i],
                        qo_per_req, kvp_per_req, kv_lens,
                    )
                    sub._qo_indptr_buf = qo_arr[i]
                    sub._paged_kv_indptr_buf = kvp_arr[i]
                    sub._paged_kv_indices_buf = kvi_arr[i]
                    sub._paged_kv_last_page_len_buf = kvl_arr[i]
                plan_hit = True
                self._plan_hits += 1

        if not plan_hit:
            self._plan_misses += 1
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
                # KV is stored at the page table's dtype (fp8_e4m3 on the
                # 70B-FP8 path), which can differ from the bf16 compute
                # dtype — plan with the store dtype or the cascade run()
                # rejects the fp8 K/V (matches paged's kv_data_type).
                kv_data_type=page_table.store_dtype,
            )
            if self.enable_plan_cache and _is_sm90(device) and plan_sig is not None:
                # Cache miss: capture the schedule's batch_indices
                # mapping per sub-wrapper for future hit patches.
                num_sm90_ctas = (
                    torch.cuda.get_device_properties(device)
                    .multi_processor_count
                )
                patches = []
                ok = True
                for sub in wrappers.cascade._batch_prefill_wrappers:
                    p = _build_sm90_patch_state(sub, num_sm90_ctas)
                    if p is None:
                        ok = False
                        break
                    patches.append(p)
                if ok:
                    self._plan_cache[id(wrappers.cascade)] = (plan_sig, patches)
        if trace_on:
            torch.cuda.synchronize()
            _t_plan = time.perf_counter()

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
        if trace_on:
            torch.cuda.synchronize()
            _t_write = time.perf_counter()
            self._plan_trace.append({
                "numpy_ms": (_t_numpy - _t0)    * 1000.0,
                "h2d_ms":   (_t_h2d   - _t_numpy) * 1000.0,
                "plan_ms":  (_t_plan  - _t_h2d) * 1000.0,
                "write_ms": (_t_write - _t_plan) * 1000.0,
            })
        return StepPlan(ctx=ctx, beam_order_per_prompt=None, pick=None)


def _dump_mlca_plan_trace(trace: list[dict], *, top_n: int = 5) -> None:
    """Print a mean/p50/p99/total breakdown of MLCA's per-step plan phases.

    Phases:
      numpy_ms : per-prompt numpy build of qo/kvp/kvi/kvl cumsum + content
      h2d_ms   : torch.from_numpy().to(device, non_blocking=True) × 8 buffers
      plan_ms  : MultiLevelCascadeAttentionWrapper.plan (= 2× BatchPrefill.plan)
      write_ms : write_pi/write_po per-beam build + 2 torch.tensor H2D
    """
    if not trace:
        print("[mlca trace] empty")
        return
    import statistics as _st
    phases = ("numpy_ms", "h2d_ms", "plan_ms", "write_ms")
    print(f"  {'phase':<12}  {'mean':>9}  {'p50':>9}  {'p99':>9}  {'total':>9}")
    print(f"  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
    for p in phases:
        xs = [e[p] for e in trace]
        xs_s = sorted(xs)
        mean = sum(xs) / len(xs)
        p50 = _st.median(xs)
        p99 = xs_s[int(0.99 * (len(xs) - 1))]
        total = sum(xs)
        print(f"  {p:<12}  {mean:9.4f}  {p50:9.4f}  {p99:9.4f}  {total:9.2f}")
    overall = [sum(e[p] for p in phases) for e in trace]
    print(f"  {'sum/step':<12}  "
          f"{sum(overall)/len(overall):9.4f}  "
          f"{_st.median(overall):9.4f}  "
          f"{sorted(overall)[int(0.99*(len(overall)-1))]:9.4f}  "
          f"{sum(overall):9.2f}")


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
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
    return_timings: bool = False,
    return_phase_timings: bool = False,
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

    The non-fused wrapper runs 2L−1 = 3 kernel launches per step (one
    attention launch per level + one LSE merge); ``adaptive_pool`` uses
    the fused wrapper that collapses the 3 launches into 1.
    """
    from ..page_driver import beam_search as _shared_beam_search

    return _shared_beam_search(
        model, config, prompt_ids, max_new_tokens, beam_width,
        backend=MlcaBackend(),
        page_size=page_size,
        max_num_pages=max_num_pages,
        device=device,
        dtype=dtype,
        kv_dtype=kv_dtype,
        return_timings=return_timings,
        return_phase_timings=return_phase_timings,
        select_at_prefill=select_at_prefill,
        select_at_decode=select_at_decode,
    )

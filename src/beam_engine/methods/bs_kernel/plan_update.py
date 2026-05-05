"""GPU-resident beam-search plan state + per-step plan-update Triton kernels.

Design plan reference: `docs/bs_kernel_design.md` § "Plan-update kernel".

Goal: eliminate the per-step Python plan-build cost the cascade method
currently pays. Today (`adaptive_pool.py` / `bs_kernel/driver.py`) every
decode step rebuilds:
  * `pages_per_beam` (Python list-of-lists)
  * `_adaptive_levels(...)` (LCA + fork-group scan in Python)
  * `_build_cascade_plan(...)` (per-level qo/kv indptr arrays as
    `torch.tensor([...], device=...)` — each call is a small CPU→GPU
    transfer)

This module replaces those steps with GPU-resident state + Triton
kernels. The state buffers persist across decode steps so per-step
work is bounded by `O(K)`, not `O(K * L_p)`.

Phase 1 status (this file):
  * `PlanState` — GPU buffers + Python orchestrator.
  * `update_pages_kernel` — parent→child page reordering after the
    beam-search topk picks parents.
  * `compute_lca_kernel` — longest-common-page-prefix across K beams.

Deferred to Phase 2 (require integration with cascade.py to consume
device tensors):
  * `compute_intermediate_groups_kernel` — uniform-fork-group detection
    for 3-level cascade.
  * `emit_cascade_arrays_kernel` — write qo_indptr_arr / kv_indptr_arr
    / kv_indices_arr / kv_last_page_len directly into device tensors
    that the wrapper's `plan()` consumes without `.cpu()` round-trips.
  * Driver integration — replace `_extract_workload` +
    `_build_cascade_plan` calls with `PlanState.step(parent_beam_ids,
    new_pages, ...)`.

Until Phase 2 lands, the existing Python path remains the source of
truth; this module's kernels are exercised only by `tests/test_bs_kernel.py`
correctness checks (when wired by the integration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass
class PlanState:
    """GPU-resident beam-search plan state.

    Layout:
      * ``pages``         [K, max_pages] int32 — per-beam page lists. Padded
                          with -1 past ``page_count[i]`` per beam.
      * ``page_count``    [K] int32 — number of valid pages per beam.
      * ``last_page_len`` [K] int32 — slot offset within the last page
                          (1..page_size).

    All buffers live on the GPU and are updated in place across decode
    steps via the Triton kernels below.
    """

    K: int
    max_pages: int
    page_size: int
    device: torch.device
    pages: torch.Tensor = field(init=False)
    page_count: torch.Tensor = field(init=False)
    last_page_len: torch.Tensor = field(init=False)
    # Scratch outputs of the Triton kernels — reused across steps to
    # avoid per-call cudaMalloc.
    lca_buf: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.pages = torch.full(
            (self.K, self.max_pages), -1,
            dtype=torch.int32, device=self.device,
        )
        self.page_count = torch.zeros(
            self.K, dtype=torch.int32, device=self.device,
        )
        self.last_page_len = torch.zeros(
            self.K, dtype=torch.int32, device=self.device,
        )
        self.lca_buf = torch.zeros(1, dtype=torch.int32, device=self.device)

    def init_from_prefill(
        self,
        prompt_pages: list[int],
        prompt_last_page_len: int,
    ) -> None:
        """Seed all K beams with the prompt's page list (post-prefill)."""
        n = len(prompt_pages)
        if n > self.max_pages:
            raise ValueError(f"prompt has {n} pages, exceeds max_pages={self.max_pages}")
        prompt_t = torch.tensor(
            prompt_pages, dtype=torch.int32, device=self.device,
        )
        # All K beams share the prompt pages.
        self.pages[:, :n].copy_(prompt_t.unsqueeze(0).expand(self.K, -1))
        self.pages[:, n:].fill_(-1)
        self.page_count.fill_(n)
        self.last_page_len.fill_(prompt_last_page_len)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def update_pages_kernel(
    pages_ptr,            # [K, max_pages] int32, in/out (one buffer flips with old_pages_ptr)
    old_pages_ptr,        # [K, max_pages] int32, in
    page_count_ptr,       # [K] int32, in/out
    old_page_count_ptr,   # [K] int32, in
    parent_beam_ids_ptr,  # [K] int32, in — parent beam index for each new beam slot
    new_page_per_beam_ptr,  # [K] int32, in — new page id appended this step
                            # (-1 sentinel when no new page allocated, e.g.
                            # mid-page write that didn't advance the page index)
    K: tl.constexpr,
    MAX_PAGES: tl.constexpr,
):
    """Reorder per-beam page lists by parent_beam_ids and append the
    new write page.

    For each new-beam slot i ∈ [0, K):
      * Read parent = parent_beam_ids[i].
      * Copy old_pages[parent, :old_page_count[parent]] → pages[i, ...].
      * If new_page_per_beam[i] >= 0, append it at position
        old_page_count[parent], advancing page_count[i] by 1; else copy
        page_count[i] = old_page_count[parent].

    Single program instance per beam (grid = K). Uses a Triton-loop over
    page positions.
    """
    pid = tl.program_id(0)
    if pid >= K:
        return

    parent = tl.load(parent_beam_ids_ptr + pid).to(tl.int32)
    old_count = tl.load(old_page_count_ptr + parent).to(tl.int32)
    new_page = tl.load(new_page_per_beam_ptr + pid).to(tl.int32)

    # Copy parent's pages to slot pid.
    offsets = tl.arange(0, MAX_PAGES)
    src_row = old_pages_ptr + parent * MAX_PAGES + offsets
    dst_row = pages_ptr + pid * MAX_PAGES + offsets
    mask_valid = offsets < old_count
    vals = tl.load(src_row, mask=mask_valid, other=-1)
    tl.store(dst_row, vals, mask=offsets < MAX_PAGES)

    # Append new page (if any).
    final_count = old_count
    if new_page >= 0:
        # Write at position old_count.
        tl.store(pages_ptr + pid * MAX_PAGES + old_count, new_page, mask=old_count < MAX_PAGES)
        final_count = old_count + 1

    tl.store(page_count_ptr + pid, final_count)


@triton.jit
def compute_lca_kernel(
    pages_ptr,        # [K, max_pages] int32
    page_count_ptr,   # [K] int32
    lca_out_ptr,      # [1] int32, out — number of fully-shared pages
    K: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Longest common page prefix across K beams.

    Each program instance scans a BLOCK of page positions. For position
    p < min(page_count), it checks whether all K beams agree on
    pages[*, p]. The LCA is the smallest p where they disagree (or
    min_count if all agree).

    Single-program-instance version (grid = 1) for now; K is small
    (≤ 256) so the in-kernel loop over beams is fine. A multi-program
    version would split the page axis and atomic-min the results.
    """
    pid = tl.program_id(0)
    if pid > 0:
        return

    # Compute min_count across beams.
    min_count = tl.load(page_count_ptr + 0).to(tl.int32)
    for k in tl.static_range(1, K):
        c = tl.load(page_count_ptr + k).to(tl.int32)
        min_count = tl.minimum(min_count, c)

    # Scan page positions until divergence.
    lca = 0
    diverged = False
    for p in range(0, MAX_PAGES):
        if (p < min_count) and (not diverged):
            pivot = tl.load(pages_ptr + 0 * MAX_PAGES + p).to(tl.int32)
            agree = True
            for k in tl.static_range(1, K):
                v = tl.load(pages_ptr + k * MAX_PAGES + p).to(tl.int32)
                if v != pivot:
                    agree = False
            if agree:
                lca = p + 1
            else:
                diverged = True

    tl.store(lca_out_ptr, lca)


# ---------------------------------------------------------------------------
# Python orchestrator
# ---------------------------------------------------------------------------


def step_update_pages(
    state: PlanState,
    parent_beam_ids: torch.Tensor,
    new_page_per_beam: torch.Tensor,
    last_page_len_per_beam: torch.Tensor,
    *,
    scratch_pages: Optional[torch.Tensor] = None,
    scratch_count: Optional[torch.Tensor] = None,
) -> None:
    """Apply one beam-search step to the plan state.

    Parameters
    ----------
    parent_beam_ids   : [K] int32 — parent beam index per new slot (from
                        the previous step's topk_flat // vocab_size).
    new_page_per_beam : [K] int32 — newly-allocated page id per beam, or
                        -1 if no new page (mid-page write).
    last_page_len_per_beam : [K] int32 — updated last-page-len per beam.

    The state buffers ``pages`` / ``page_count`` are updated in place.
    A scratch buffer pair is needed because we read from the old slot
    while writing to the new slot; the caller can pass pre-allocated
    scratch tensors to amortize.
    """
    K = state.K
    MAX_PAGES = state.max_pages

    if scratch_pages is None:
        scratch_pages = state.pages.clone()
    else:
        scratch_pages.copy_(state.pages)
    if scratch_count is None:
        scratch_count = state.page_count.clone()
    else:
        scratch_count.copy_(state.page_count)

    update_pages_kernel[(K,)](
        state.pages,
        scratch_pages,
        state.page_count,
        scratch_count,
        parent_beam_ids,
        new_page_per_beam,
        K=K,
        MAX_PAGES=MAX_PAGES,
    )
    state.last_page_len.copy_(last_page_len_per_beam)


def compute_lca(state: PlanState) -> torch.Tensor:
    """Run the LCA Triton kernel; returns the [1]-int32 result tensor.

    The result is a device tensor — caller can use `.item()` if a host
    int is needed, or pass it directly to a downstream Triton kernel
    that consumes the LCA depth.
    """
    compute_lca_kernel[(1,)](
        state.pages,
        state.page_count,
        state.lca_buf,
        K=state.K,
        MAX_PAGES=state.max_pages,
        BLOCK=state.max_pages,
    )
    return state.lca_buf

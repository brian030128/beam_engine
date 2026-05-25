"""Baseline 1 — Paged-attention beam search.

Each beam is treated as an independent sequence by FlashInfer's paged decode
wrapper. Prefix KV pages are shared across beams via a refcount; copy-on-write
runs at divergence; pruned beams release page refs. The shared prefix lives in
HBM exactly once but is read once per beam at decode time
(``O(B * L_p)`` traffic per step).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import flashinfer.page
import numpy as np
import torch
import torch.nn.functional as F
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
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
# Attention context — dispatches q/k/v through FlashInfer paged wrappers
# ---------------------------------------------------------------------------


@dataclass
class PagedAttentionContext(AttentionContext):
    """Per-forward state for FlashInfer paged attention.

    The driver builds this once before each forward, then the model's per-layer
    attention call routes through ``attend``. Set ``is_prefill=True`` to use
    the prefill wrapper (ragged-batch, causal) or False for the decode wrapper.
    """

    is_prefill: bool
    page_table: PageTable
    kv_page_indices: torch.Tensor      # [nnz] int32 — page idx per token to write
    kv_page_offsets: torch.Tensor      # [nnz] int32 — offset within page per token
    prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = None
    decode_wrapper: BatchDecodeWithPagedKVCacheWrapper | None = None
    _write_helper_indptr: torch.Tensor | None = field(default=None, repr=False)

    def _get_kv_write_helpers(self, nnz: int, device: torch.device):
        buf = self._write_helper_indptr
        if buf is None or buf.shape[0] < nnz + 1:
            buf = torch.arange(nnz + 1, dtype=torch.int32, device=device)
            self._write_helper_indptr = buf
        return buf[:nnz], buf[: nnz + 1]

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        num_kv_heads = self.page_table.head_num
        head_dim = self.page_table.head_dim
        num_heads = q.shape[-1] // head_dim

        kv_cache = self.page_table.kv_cache_at_layer[layer_idx]
        # kv_cache layout: [2, max_pages, page_size, num_kv_heads, head_dim].
        # Pass (k, v) tuple so wrappers / append_paged_kv_cache get
        # contiguous slabs without an unbind+strided view.
        kv_tuple = (kv_cache[0], kv_cache[1])

        k_3d = k.reshape(-1, num_kv_heads, head_dim)
        v_3d = v.reshape(-1, num_kv_heads, head_dim)
        # If KV cache is stored at a narrower dtype than the activation
        # dtype (fp8 KV with bf16 compute on Llama-3-70B-FP8), cast the
        # appended K/V into the storage dtype. The attention kernel
        # handles the fp8→bf16 dequant on read internally (planned with
        # kv_data_type=fp8).
        if k_3d.dtype != kv_cache.dtype:
            k_3d = k_3d.to(kv_cache.dtype)
            v_3d = v_3d.to(kv_cache.dtype)
        batch_indices, kv_indptr = self._get_kv_write_helpers(k_3d.shape[0], k_3d.device)
        flashinfer.page.append_paged_kv_cache(
            append_key=k_3d,
            append_value=v_3d,
            batch_indices=batch_indices,
            positions=self.kv_page_offsets,
            paged_kv_cache=kv_tuple,
            kv_indices=self.kv_page_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=self.kv_page_offsets,
            kv_layout="NHD",
        )

        q_3d = q.reshape(-1, num_heads, head_dim)
        if self.is_prefill:
            output = self.prefill_wrapper.run(q_3d, kv_tuple)
        else:
            output = self.decode_wrapper.run(q_3d, kv_tuple)
        return output.reshape(*q.shape[:-1], num_heads * head_dim)


# ---------------------------------------------------------------------------
# Beam state
# ---------------------------------------------------------------------------


@dataclass
class Beam:
    token_ids: list[int]       # generated tokens (excluding prompt)
    cum_log_prob: float
    pages: list[int]           # ordered physical page indices for this beam


def _add_ref(rc: dict[int, int], page: int, count: int = 1) -> None:
    rc[page] = rc.get(page, 0) + count


def _remove_ref(rc: dict[int, int], page: int, page_table: PageTable, count: int = 1) -> None:
    rc[page] -= count
    assert rc[page] >= 0
    if rc[page] == 0:
        page_table.free_block(page)
        del rc[page]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


from dataclasses import dataclass as _dataclass


class _PagedWrappers:
    __slots__ = ("decode_wrapper",)


@_dataclass
class PagedBackend:
    name: str = "paged"
    use_split_pages: bool = False  # paged doesn't model "shared prefix" the way cascade does

    # Sub-phase trace for plan_decode_step. Enabled by env var
    # PAGED_TRACE_PLAN=1. Each entry is a dict of per-phase ms timings
    # for one step.
    _plan_trace: list = field(default_factory=list, init=False, repr=False)

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
        wb = _PagedWrappers()
        wb.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout="NHD", use_tensor_cores=True,
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
        import time as _time
        trace_on = bool(int(_os.environ.get("PAGED_TRACE_PLAN", "0")))
        _verify_lca = bool(int(_os.environ.get("PAGED_VERIFY_LCA", "0")))
        if trace_on:
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()
        ps = page_size
        # B*K independent paged-decode sequences in one launch.
        #
        # Two-pass numpy build instead of the legacy
        # ``torch.tensor(python_list, dtype=int32)`` path:
        #   pass 1 — compute per-beam pages-length to size the indices
        #            buffer + fill the scalar arrays (indptr, lpl,
        #            write_pi, write_po)
        #   pass 2 — slice-assign each beam's pages list into the
        #            pre-allocated numpy buffer (numpy converts PyObject
        #            ints in C, ~30x faster than torch.tensor on a
        #            multi-million-entry Python list).
        # Then a single ``torch.from_numpy().to(device, non_blocking=True)``
        # per tensor lets the 5 small H2D copies overlap. Mirrors the
        # numpy-bridge pattern bs_kernel uses for its decode-wrapper plan.
        n_beams = B * K
        indptr_np = np.empty(n_beams + 1, dtype=np.int32)
        indptr_np[0] = 0
        lpl_np = np.empty(n_beams, dtype=np.int32)
        write_pi_np = np.empty(n_beams, dtype=np.int32)
        write_po_np = np.empty(n_beams, dtype=np.int32)
        row = 0
        for b in range(B):
            pos = current_pos[b]
            off = pos % ps
            pli = pos // ps
            for beam in beams_per_prompt[b]:
                indptr_np[row + 1] = indptr_np[row] + len(beam.pages)
                lpl_np[row] = off + 1
                write_pi_np[row] = beam.pages[pli]
                write_po_np[row] = off
                row += 1
        if trace_on:
            _t_pass1 = _time.perf_counter()
        total = int(indptr_np[n_beams])
        indices_np = np.empty(total, dtype=np.int32)
        row = 0
        for b in range(B):
            beams = beams_per_prompt[b]
            # LCA-prefix-skip for the index build: all K beams of a prompt
            # share identical prefix pages — the prefill prefix (length
            # ``last_lca_per_prompt[b]``) is refcounted and never rewritten
            # during decode (CoW only touches the active tail page). So
            # convert that prefix Python-list→numpy ONCE per prompt and
            # memcpy it into every beam's slot (numpy→numpy), instead of
            # re-converting it K times. On long-prefix cells this prefix is
            # the bulk of the index array (e.g. ~5000 of ~5016 pages/beam at
            # L_p=80K), so this removes ~K× of the per-step build cost. The
            # emitted indices are bit-identical to the per-beam build.
            lca = last_lca_per_prompt[b]
            prefix_arr = (
                np.asarray(beams[0].pages[:lca], dtype=np.int32)
                if lca else None
            )
            for beam in beams:
                start = indptr_np[row]
                end = indptr_np[row + 1]
                if prefix_arr is not None:
                    indices_np[start:start + lca] = prefix_arr
                    if end > start + lca:
                        indices_np[start + lca:end] = beam.pages[lca:]
                else:
                    indices_np[start:end] = beam.pages
                if _verify_lca:
                    # Prove the dedup matches the naive per-beam build
                    # (the shared-prefix invariant holds). Raises if not.
                    assert indices_np[start:end].tolist() == list(beam.pages), (
                        f"LCA dedup mismatch at prompt {b} (lca={lca}): "
                        f"prefix not shared across beams"
                    )
                row += 1
        if trace_on:
            _t_pass2 = _time.perf_counter()

        indptr_t = torch.from_numpy(indptr_np).to(device, non_blocking=True)
        indices_t = torch.from_numpy(indices_np).to(device, non_blocking=True)
        lpl_t = torch.from_numpy(lpl_np).to(device, non_blocking=True)
        write_pi_t = torch.from_numpy(write_pi_np).to(device, non_blocking=True)
        write_po_t = torch.from_numpy(write_po_np).to(device, non_blocking=True)
        if trace_on:
            torch.cuda.synchronize()
            _t_h2d = _time.perf_counter()

        wrappers.decode_wrapper.plan(
            indptr=indptr_t,
            indices=indices_t,
            last_page_len=lpl_t,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=ps,
            q_data_type=dtype,
            kv_data_type=page_table.store_dtype,
        )
        if trace_on:
            torch.cuda.synchronize()
            _t_plan = _time.perf_counter()
            self._plan_trace.append({
                "pass1_ms":     (_t_pass1 - _t0) * 1000.0,
                "pass2_ms":     (_t_pass2 - _t_pass1) * 1000.0,
                "h2d_ms":       (_t_h2d   - _t_pass2) * 1000.0,
                "plan_call_ms": (_t_plan  - _t_h2d) * 1000.0,
            })
        ctx = PagedAttentionContext(
            is_prefill=False,
            page_table=page_table,
            kv_page_indices=write_pi_t,
            kv_page_offsets=write_po_t,
            decode_wrapper=wrappers.decode_wrapper,
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
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
    return_timings: bool = False,
    return_phase_timings: bool = False,
    select_at_prefill: PrefillSelect = standard_prefill_select,
    select_at_decode: DecodeSelect = standard_decode_select,
):
    """Batched beam search with copy-on-write paged KV cache.

    Returns ``list[list[Beam]]`` — outer index per prompt, inner sorted by
    ``cum_log_prob`` (best first). When ``return_timings=True``, returns
    ``(beams, timings)`` where timings is ``{prefill_ms, decode_step_ms: list}``.

    ``select_at_prefill`` / ``select_at_decode`` plug in alternate top-K
    strategies (see ``beam_engine.decoding``); defaults are standard top-K.

    ``kv_dtype`` (default = ``dtype``) stores the page-table K/V at a
    narrower dtype for fp8-KV runs (Llama-3-70B-FP8 path).
    """
    from ..page_driver import beam_search as _shared_beam_search

    backend = PagedBackend()
    result = _shared_beam_search(
        model, config, prompt_ids, max_new_tokens, beam_width,
        backend=backend,
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
    import os as _os
    if int(_os.environ.get("PAGED_TRACE_PLAN", "0")):
        _dump_paged_plan_trace(backend._plan_trace)
    return result


def _dump_paged_plan_trace(trace: list) -> None:
    """Print sub-phase summary of paged plan_decode_step."""
    import statistics
    if not trace:
        print("[paged plan-trace] no entries")
        return
    n = len(trace)
    fields = ("pass1_ms", "pass2_ms", "h2d_ms", "plan_call_ms")
    print(f"\n[paged plan-trace] {n} steps")
    print(f"  {'phase':<14} {'mean':>10} {'median':>10} {'p90':>10} {'p99':>10} {'total':>12}")
    for f in fields:
        xs = [r[f] for r in trace]
        xs_sorted = sorted(xs)
        mean = sum(xs) / n
        med = statistics.median(xs)
        p90 = xs_sorted[int(0.9 * (n - 1))]
        p99 = xs_sorted[int(0.99 * (n - 1))]
        total = sum(xs)
        print(
            f"  {f:<14} {mean:>10.3f} {med:>10.3f} "
            f"{p90:>10.3f} {p99:>10.3f} {total:>12.2f}"
        )



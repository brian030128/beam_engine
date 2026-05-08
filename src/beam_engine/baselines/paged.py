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

        k_3d = k.view(-1, num_kv_heads, head_dim)
        v_3d = v.view(-1, num_kv_heads, head_dim)
        batch_indices, kv_indptr = self._get_kv_write_helpers(k_3d.shape[0], k_3d.device)
        flashinfer.page.append_paged_kv_cache(
            append_key=k_3d,
            append_value=v_3d,
            batch_indices=batch_indices,
            positions=self.kv_page_offsets,
            paged_kv_cache=kv_cache,
            kv_indices=self.kv_page_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=self.kv_page_offsets,
            kv_layout="NHD",
        )

        q_3d = q.view(-1, num_heads, head_dim)
        if self.is_prefill:
            output = self.prefill_wrapper.run(q_3d, kv_cache)
        else:
            output = self.decode_wrapper.run(q_3d, kv_cache)
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

        ps = page_size
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
        wrappers.decode_wrapper.plan(
            indptr=torch.tensor(flat_indptr, dtype=torch.int32, device=device),
            indices=torch.tensor(flat_indices, dtype=torch.int32, device=device),
            last_page_len=torch.tensor(flat_lpl, dtype=torch.int32, device=device),
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
    return_timings: bool = False,
    select_at_prefill: PrefillSelect = standard_prefill_select,
    select_at_decode: DecodeSelect = standard_decode_select,
):
    """Batched beam search with copy-on-write paged KV cache.

    Returns ``list[list[Beam]]`` — outer index per prompt, inner sorted by
    ``cum_log_prob`` (best first). When ``return_timings=True``, returns
    ``(beams, timings)`` where timings is ``{prefill_ms, decode_step_ms: list}``.

    ``select_at_prefill`` / ``select_at_decode`` plug in alternate top-K
    strategies (see ``beam_engine.decoding``); defaults are standard top-K.
    """
    from ..page_driver import beam_search as _shared_beam_search

    return _shared_beam_search(
        model, config, prompt_ids, max_new_tokens, beam_width,
        backend=PagedBackend(),
        page_size=page_size,
        max_num_pages=max_num_pages,
        device=device,
        dtype=dtype,
        return_timings=return_timings,
        select_at_prefill=select_at_prefill,
        select_at_decode=select_at_decode,
    )



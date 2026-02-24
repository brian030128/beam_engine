from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import flashinfer.page


@dataclass
class AttentionMetadata:
    is_prefill: bool
    prefill_wrapper: Any | None = None  # flashinfer BatchPrefillWithPagedKVCacheWrapper
    decode_wrapper: Any | None = None   # flashinfer decode wrapper (BatchDecode or MultiLevelCascade)
    page_table: Any | None = None       # PageTable instance
    kv_page_indices: torch.Tensor | None = None  # [nnz] int32 — physical page for each token to write
    kv_page_offsets: torch.Tensor | None = None   # [nnz] int32 — offset within page for each token to write


class FlashInferAttention(nn.Module):
    """FlashInfer paged attention layer."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        layer_idx: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_idx = layer_idx

    def _get_kv_write_helpers(self, nnz: int, device: torch.device):
        """Return (batch_indices, kv_indptr) for append_paged_kv_cache, cached."""
        buf = getattr(self, "_kv_write_indptr", None)
        if buf is None or buf.shape[0] < nnz + 1:
            buf = torch.arange(nnz + 1, dtype=torch.int32, device=device)
            self._kv_write_indptr = buf
        return buf[:nnz], buf[:nnz + 1]  # batch_indices, kv_indptr

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if attn_metadata.is_prefill:
            return self._flashinfer_prefill(q, k, v, attn_metadata)
        else:
            return self._flashinfer_decode(q, k, v, attn_metadata)

    def _flashinfer_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        page_table = attn_metadata.page_table
        kv_cache = page_table.kv_cache_at_layer[self.layer_idx]

        # Write k/v to paged cache via fused FlashInfer kernel
        k_3d = k.view(-1, self.num_kv_heads, self.head_dim)
        v_3d = v.view(-1, self.num_kv_heads, self.head_dim)
        pi = attn_metadata.kv_page_indices
        po = attn_metadata.kv_page_offsets
        batch_indices, kv_indptr = self._get_kv_write_helpers(k_3d.shape[0], k_3d.device)
        flashinfer.page.append_paged_kv_cache(
            append_key=k_3d,
            append_value=v_3d,
            batch_indices=batch_indices,
            positions=po,
            paged_kv_cache=kv_cache,
            kv_indices=pi,
            kv_indptr=kv_indptr,
            kv_last_page_len=po,
            kv_layout='NHD',
        )

        # Attention against full paged cache
        q_3d = q.view(-1, self.num_heads, self.head_dim)
        output = attn_metadata.prefill_wrapper.run(q_3d, kv_cache)
        return output.reshape(*q.shape[:-1], self.num_heads * self.head_dim)

    def _flashinfer_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        page_table = attn_metadata.page_table
        kv_cache = page_table.kv_cache_at_layer[self.layer_idx]

        # Write k/v to paged cache via fused FlashInfer kernel
        k_3d = k.view(-1, self.num_kv_heads, self.head_dim)
        v_3d = v.view(-1, self.num_kv_heads, self.head_dim)
        pi = attn_metadata.kv_page_indices
        po = attn_metadata.kv_page_offsets
        batch_indices, kv_indptr = self._get_kv_write_helpers(k_3d.shape[0], k_3d.device)
        flashinfer.page.append_paged_kv_cache(
            append_key=k_3d,
            append_value=v_3d,
            batch_indices=batch_indices,
            positions=po,
            paged_kv_cache=kv_cache,
            kv_indices=pi,
            kv_indptr=kv_indptr,
            kv_last_page_len=po,
            kv_layout='NHD',
        )

        # Attention against full paged cache
        q_3d = q.view(-1, self.num_heads, self.head_dim)
        output = attn_metadata.decode_wrapper.run(q_3d, kv_cache)
        return output.reshape(*q.shape[:-1], self.num_heads * self.head_dim)

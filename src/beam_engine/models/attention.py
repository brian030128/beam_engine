from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionMetadata:
    is_prefill: bool
    prefill_wrapper: Any | None = None  # flashinfer BatchPrefillWithPagedKVCacheWrapper
    decode_wrapper: Any | None = None   # flashinfer BatchDecodeWithPagedKVCacheWrapper
    page_table: Any | None = None       # PageTable instance
    kv_page_indices: torch.Tensor | None = None  # [nnz] int32 — physical page for each token to write
    kv_page_offsets: torch.Tensor | None = None   # [nnz] int32 — offset within page for each token to write


class FlashInferAttention(nn.Module):
    """Attention layer with naive SDPA fallback and flashinfer paged attention."""

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
        self.num_kv_groups = num_heads // num_kv_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return self._naive_attention(q, k, v)

        # Phase 2: flashinfer paged attention
        if attn_metadata.is_prefill and attn_metadata.prefill_wrapper is not None:
            return self._flashinfer_prefill(q, k, v, attn_metadata)
        elif not attn_metadata.is_prefill and attn_metadata.decode_wrapper is not None:
            return self._flashinfer_decode(q, k, v, attn_metadata)

        return self._naive_attention(q, k, v)

    def _naive_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # q: [..., num_heads * head_dim], k/v: [..., num_kv_heads * head_dim]
        batch_dims = q.shape[:-1]

        q = q.view(*batch_dims, self.num_heads, self.head_dim)
        k = k.view(*batch_dims, self.num_kv_heads, self.head_dim)
        v = v.view(*batch_dims, self.num_kv_heads, self.head_dim)

        # GQA: expand k/v heads to match q heads
        # [batch, seq, num_kv_heads, head_dim] → [batch, seq, num_kv_heads, num_kv_groups, head_dim]
        # → [batch, seq, num_heads, head_dim]
        if self.num_kv_groups > 1:
            # [B, S, num_kv_heads, head_dim] → [B, S, num_kv_heads, 1, head_dim]
            # → [B, S, num_kv_heads, num_kv_groups, head_dim] → [B, S, num_heads, head_dim]
            k = k.unsqueeze(-2).expand(*k.shape[:-1], self.num_kv_groups, k.shape[-1]).reshape(*batch_dims, self.num_heads, self.head_dim)
            v = v.unsqueeze(-2).expand(*v.shape[:-1], self.num_kv_groups, v.shape[-1]).reshape(*batch_dims, self.num_heads, self.head_dim)

        # SDPA expects [batch, num_heads, seq_len, head_dim]
        # Input is [batch, seq_len, num_heads, head_dim]
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Transpose back to [batch, seq_len, num_heads, head_dim] and flatten
        attn_output = attn_output.transpose(-3, -2)
        attn_output = attn_output.reshape(*batch_dims, self.num_heads * self.head_dim)
        return attn_output

    def _flashinfer_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        page_table = attn_metadata.page_table
        kv_cache = page_table.kv_cache_at_layer[self.layer_idx]

        # Write k/v to paged cache
        # kv_cache shape: [max_num_pages, 2, page_size, num_kv_heads, head_dim]
        k_3d = k.view(-1, self.num_kv_heads, self.head_dim)
        v_3d = v.view(-1, self.num_kv_heads, self.head_dim)
        pi = attn_metadata.kv_page_indices
        po = attn_metadata.kv_page_offsets
        kv_cache[pi, 0, po] = k_3d
        kv_cache[pi, 1, po] = v_3d

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

        # Write k/v to paged cache
        # kv_cache shape: [max_num_pages, 2, page_size, num_kv_heads, head_dim]
        k_3d = k.view(-1, self.num_kv_heads, self.head_dim)
        v_3d = v.view(-1, self.num_kv_heads, self.head_dim)
        pi = attn_metadata.kv_page_indices
        po = attn_metadata.kv_page_offsets
        kv_cache[pi, 0, po] = k_3d
        kv_cache[pi, 1, po] = v_3d

        # Attention against full paged cache
        q_3d = q.view(-1, self.num_heads, self.head_dim)
        output = attn_metadata.decode_wrapper.run(q_3d, kv_cache)
        return output.reshape(*q.shape[:-1], self.num_heads * self.head_dim)

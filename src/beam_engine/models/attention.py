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


class FlashInferAttention(nn.Module):
    """Attention layer with naive SDPA fallback and future flashinfer paged attention."""

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
        raise NotImplementedError("FlashInfer prefill not yet implemented (Phase 2)")

    def _flashinfer_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError("FlashInfer decode not yet implemented (Phase 2)")

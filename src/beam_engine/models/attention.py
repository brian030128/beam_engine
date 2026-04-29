"""Attention dispatch layer.

The Llama model calls ``backend.attend(q, k, v, layer_idx)``. The backend is
supplied per forward by an ``AttentionContext`` object — one implementation per
baseline (paged, tree). Backends own the KV cache storage and the FlashInfer /
FlashAttention call, so the model code is identical across baselines.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class AttentionContext(Protocol):
    """Per-forward attention state and dispatch."""

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        ...


class Attention(nn.Module):
    """Stateless attention call site — just forwards to the active backend."""

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttentionContext,
    ) -> torch.Tensor:
        return ctx.attend(q, k, v, self.layer_idx)

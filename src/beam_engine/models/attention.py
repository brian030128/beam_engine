"""Attention dispatch layer.

The Llama model calls ``backend.attend(q, k, v, layer_idx)``. The backend is
supplied per forward by an ``AttentionContext`` object — one implementation per
baseline (paged, tree). Backends own the KV cache storage and the FlashInfer /
FlashAttention call, so the model code is identical across baselines.
"""

from __future__ import annotations

import os
import time as _time
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

# Optional kernel-time instrumentation. Enable by setting BE_PROFILE_ATTN=1.
_PROFILE = os.environ.get("BE_PROFILE_ATTN") == "1"
_ATTN_TIMES = {"attn_ms": 0.0}


def get_attn_subtimes() -> dict[str, float]:
    return dict(_ATTN_TIMES)


def reset_attn_subtimes() -> None:
    _ATTN_TIMES["attn_ms"] = 0.0


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
        if _PROFILE:
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            out = ctx.attend(q, k, v, self.layer_idx)
            torch.cuda.synchronize()
            _ATTN_TIMES["attn_ms"] += (_time.perf_counter() - t0) * 1000.0
            return out
        return ctx.attend(q, k, v, self.layer_idx)

import math

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Pure PyTorch rotary position embedding with LLaMA 3.1 scaling support."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        num_heads: int = 1,
        num_kv_heads: int = 1,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )

        if rope_scaling is not None:
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
            if rope_type == "llama3":
                inv_freq = self._apply_llama3_scaling(inv_freq, rope_scaling)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0

    def _apply_llama3_scaling(
        self, inv_freq: torch.Tensor, rope_scaling: dict
    ) -> torch.Tensor:
        factor = rope_scaling["factor"]
        low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
        old_context_len = rope_scaling.get(
            "original_max_position_embeddings", 8192
        )

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelens = 2 * math.pi / inv_freq
        new_inv_freq = []
        for freq, wavelen in zip(inv_freq, wavelens):
            if wavelen < high_freq_wavelen:
                # High frequency — no scaling
                new_inv_freq.append(freq)
            elif wavelen > low_freq_wavelen:
                # Low frequency — full scaling
                new_inv_freq.append(freq / factor)
            else:
                # Medium frequency — smooth interpolation
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_inv_freq.append(
                    (1 - smooth) * freq / factor + smooth * freq
                )

        return torch.tensor(new_inv_freq, dtype=inv_freq.dtype)

    def _update_cos_sin_cache(self, max_seq_len: int, device: torch.device, dtype: torch.dtype):
        if max_seq_len <= self._cached_seq_len and self._cos_cached is not None:
            return
        self._cached_seq_len = max_seq_len
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_pos = positions.max().item() + 1
        self._update_cos_sin_cache(int(max_pos), positions.device, q.dtype)

        # Reshape packed q/k to [..., num_heads, head_dim]
        q_shape = q.shape
        k_shape = k.shape
        q = q.view(*q_shape[:-1], self.num_heads, self.head_dim)
        k = k.view(*k_shape[:-1], self.num_kv_heads, self.head_dim)

        # Gather cos/sin for the given positions
        cos = self._cos_cached[positions.long()]  # [..., head_dim]
        sin = self._sin_cached[positions.long()]

        # Expand for heads dimension
        if cos.dim() < q.dim():
            cos = cos.unsqueeze(-2)  # [..., 1, head_dim]
            sin = sin.unsqueeze(-2)

        q_rotated = _apply_rotary(q, cos, sin)
        k_rotated = _apply_rotary(k, cos, sin)

        # Reshape back to packed format
        q_rotated = q_rotated.reshape(q_shape)
        k_rotated = k_rotated.reshape(k_shape)
        return q_rotated, k_rotated


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin

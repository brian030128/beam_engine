import torch
import torch.nn as nn
from flashinfer.rope import apply_rope_pos_ids, apply_llama31_rope_pos_ids


class RotaryEmbedding(nn.Module):
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

        if rope_scaling is not None:
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        else:
            rope_type = "default"

        if rope_type == "llama3":
            self._rope_fn = apply_llama31_rope_pos_ids
            self._rope_kwargs = dict(
                rope_scale=rope_scaling["factor"],
                rope_theta=rope_theta,
                low_freq_factor=rope_scaling.get("low_freq_factor", 1.0),
                high_freq_factor=rope_scaling.get("high_freq_factor", 4.0),
                old_context_len=rope_scaling.get("original_max_position_embeddings", 8192),
            )
        else:
            self._rope_fn = apply_rope_pos_ids
            self._rope_kwargs = dict(
                rope_theta=rope_theta,
            )

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_shape = q.shape
        k_shape = k.shape
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        pos_ids = positions.view(-1)

        q, k = self._rope_fn(q, k, pos_ids, **self._rope_kwargs)

        return q.reshape(q_shape), k.reshape(k_shape)

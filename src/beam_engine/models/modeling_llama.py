"""Inference-only LLaMA model — standalone PyTorch, no vLLM dependencies."""

from __future__ import annotations

import glob
import os

import flashinfer.activation
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

from .rmsnorm import RMSNorm
from .rotary_embedding import RotaryEmbedding
from .attention import FlashInferAttention, AttentionMetadata


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        return self.down_proj(flashinfer.activation.silu_and_mul(gate_up))


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(config, "head_dim", None) or (self.hidden_size // self.num_heads)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        bias = getattr(config, "attention_bias", False)
        self.qkv_proj = nn.Linear(
            self.hidden_size, self.q_size + 2 * self.kv_size, bias=bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            rope_theta=getattr(config, "rope_theta", 10000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )
        self.attn = FlashInferAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, attn_metadata)
        return self.o_proj(attn_output)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=getattr(config, "mlp_bias", False),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, attn_metadata)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, attn_metadata)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, attn_metadata)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
    ) -> LlamaForCausalLM:
        config = LlamaConfig.from_pretrained(model_name)

        # Build model on meta device (no memory)
        with torch.device("meta"):
            model = cls(config)

        # Download safetensors
        model_dir = snapshot_download(
            model_name,
            allow_patterns=["*.safetensors", "*.json"],
        )

        # Load all safetensors shards into a single state dict
        safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        raw_state_dict: dict[str, torch.Tensor] = {}
        for f in safetensor_files:
            raw_state_dict.update(load_file(f, device="cpu"))

        # Remap weights
        remapped = _remap_state_dict(raw_state_dict, config)

        # Load into model — assign=True replaces meta tensors in-place
        model.load_state_dict(remapped, strict=True, assign=True)
        model.to(dtype=dtype, device=device)
        model.eval()
        return model


def _remap_state_dict(
    raw: dict[str, torch.Tensor], config: LlamaConfig
) -> dict[str, torch.Tensor]:
    """Remap HuggingFace LLaMA weights to our fused projection format."""
    remapped: dict[str, torch.Tensor] = {}

    # Collect per-layer q/k/v and gate/up for concatenation, keyed by (layer, suffix)
    # e.g. qkv_parts[(5, "weight")] = {"q_proj": tensor, "k_proj": tensor, "v_proj": tensor}
    qkv_parts: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    gate_up_parts: dict[tuple[int, str], dict[str, torch.Tensor]] = {}

    for name, tensor in raw.items():
        # Skip non-parameter keys
        if "rotary_emb.inv_freq" in name:
            continue
        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            continue

        # Detect q/k/v proj weights and biases
        matched = False
        for proj in ("q_proj", "k_proj", "v_proj"):
            key = f"self_attn.{proj}"
            if key in name:
                layer_idx = _extract_layer_idx(name)
                suffix = name.rsplit(".", 1)[-1]  # "weight" or "bias"
                dict_key = (layer_idx, suffix)
                if dict_key not in qkv_parts:
                    qkv_parts[dict_key] = {}
                qkv_parts[dict_key][proj] = tensor
                matched = True
                break

        if matched:
            continue

        # Detect gate/up proj weights and biases
        for proj in ("gate_proj", "up_proj"):
            key = f"mlp.{proj}"
            if key in name:
                layer_idx = _extract_layer_idx(name)
                suffix = name.rsplit(".", 1)[-1]
                dict_key = (layer_idx, suffix)
                if dict_key not in gate_up_parts:
                    gate_up_parts[dict_key] = {}
                gate_up_parts[dict_key][proj] = tensor
                matched = True
                break

        if matched:
            continue

        # Direct mapping
        remapped[name] = tensor

    # Fuse q+k+v → qkv_proj (for both weight and bias if present)
    for (layer_idx, suffix), parts in qkv_parts.items():
        fused = torch.cat([parts["q_proj"], parts["k_proj"], parts["v_proj"]], dim=0)
        remapped[f"model.layers.{layer_idx}.self_attn.qkv_proj.{suffix}"] = fused

    # Fuse gate+up → gate_up_proj (for both weight and bias if present)
    for (layer_idx, suffix), parts in gate_up_parts.items():
        fused = torch.cat([parts["gate_proj"], parts["up_proj"]], dim=0)
        remapped[f"model.layers.{layer_idx}.mlp.gate_up_proj.{suffix}"] = fused

    # Tied embeddings: checkpoint omits lm_head.weight when tie_word_embeddings=True
    if getattr(config, "tie_word_embeddings", False):
        embed_key = "model.embed_tokens.weight"
        head_key = "lm_head.weight"
        if embed_key in remapped and head_key not in remapped:
            remapped[head_key] = remapped[embed_key]

    return remapped


def _extract_layer_idx(name: str) -> int:
    """Extract layer index from a weight name like 'model.layers.5.self_attn.q_proj.weight'."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    raise ValueError(f"Cannot extract layer index from: {name}")

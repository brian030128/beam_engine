"""Inference-only Qwen3 model — standalone PyTorch, no vLLM dependencies.

Mirrors ``modeling_llama.py`` in structure (same SwiGLU MLP, same RMSNorm,
same RoPE, same KV-cache layout via ``AttentionContext``) with one
architectural delta: Qwen3 applies per-head RMSNorm to Q and K after the
QKV linear and **before** RoPE (``self_attn.q_norm`` / ``self_attn.k_norm``,
each of shape ``(head_dim,)``). Everything else is identical to Llama, so
the six beam-search backends drop in unchanged.

Supports the same tensor-parallel layout as Llama via
``beam_engine.distributed``.
"""

from __future__ import annotations

import glob
import os

import flashinfer.activation
import torch
import torch.nn as nn
from transformers import Qwen3Config
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

from .rmsnorm import RMSNorm
from .rotary_embedding import RotaryEmbedding
from .attention import Attention, AttentionContext
from ..distributed import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    get_tp_world_size,
)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], bias=bias
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        return self.down_proj(flashinfer.activation.silu_and_mul(gate_up))


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        tp_size = get_tp_world_size()
        self.hidden_size = config.hidden_size
        self.num_heads_total = config.num_attention_heads
        self.num_kv_heads_total = getattr(
            config, "num_key_value_heads", self.num_heads_total
        )
        self.head_dim = getattr(config, "head_dim", None) or (
            self.hidden_size // self.num_heads_total
        )

        self.num_heads = self.num_heads_total // tp_size
        self.num_kv_heads = self.num_kv_heads_total // tp_size

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        bias = getattr(config, "attention_bias", False)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads_total,
            self.num_kv_heads_total,
            bias=bias,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads_total * self.head_dim, self.hidden_size, bias=bias
        )

        # Per-head Q/K RMSNorm applied before RoPE — Qwen3's only meaningful
        # delta vs Llama. RMSNorm here normalizes across head_dim (the last
        # dim after reshape to (..., num_heads, head_dim)).
        rms_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_eps)

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            rope_theta=getattr(config, "rope_theta", 10000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )
        self.attn = Attention(layer_idx=layer_idx)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: AttentionContext,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head RMSNorm on q and k before RoPE. Preserve the original
        # (..., q_size) / (..., kv_size) shape so downstream callers see the
        # same layout as Llama. Use reshape (not view) because q/k after
        # split share strides with qkv and the merge to (-1, head_dim)
        # inside RMSNorm would fail a strict view.
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(q_shape)
        k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(k_shape)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, ctx)
        return self.o_proj(attn_output)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(
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
        ctx: AttentionContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, ctx)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: AttentionContext,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, ctx)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: AttentionContext,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, ctx)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
    ) -> "Qwen3ForCausalLM":
        config = Qwen3Config.from_pretrained(model_name)

        with torch.device("meta"):
            model = cls(config)

        model_dir = snapshot_download(
            model_name,
            allow_patterns=["*.safetensors", "*.json"],
        )

        safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        raw_state_dict: dict[str, torch.Tensor] = {}
        for f in safetensor_files:
            raw_state_dict.update(load_file(f, device="cpu"))

        remapped = _remap_state_dict(raw_state_dict, config)

        if "lm_head.weight" not in remapped and "model.embed_tokens.weight" in remapped:
            remapped["lm_head.weight"] = remapped["model.embed_tokens.weight"]

        _load_into_model(model, remapped, dtype=dtype, device=device)
        model.eval()
        return model


def _load_into_model(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
    device: str | torch.device,
) -> None:
    """Same materialization path as the Llama loader (meta-init →
    per-parameter materialize, routing TP-sharded params through their
    ``weight_loader`` / ``bias_loader``)."""
    tp_layers = (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )

    tp_module_prefixes: dict[str, nn.Module] = {}
    for full_name, module in model.named_modules():
        if isinstance(module, tp_layers):
            tp_module_prefixes[full_name] = module

    def _materialize(param_path: str, value: torch.Tensor) -> None:
        parent_path, _, attr = param_path.rpartition(".")
        parent: nn.Module = model.get_submodule(parent_path) if parent_path else model
        is_bias = attr == "bias"

        owner_prefix = None
        for prefix in tp_module_prefixes:
            if param_path == f"{prefix}.weight" or param_path == f"{prefix}.bias":
                owner_prefix = prefix
                break

        if owner_prefix is not None:
            tp_mod = tp_module_prefixes[owner_prefix]
            target_param = getattr(tp_mod, attr)
            new_param = nn.Parameter(
                torch.empty_like(target_param, device=device, dtype=dtype),
                requires_grad=False,
            )
            setattr(tp_mod, attr, new_param)
            if is_bias:
                tp_mod.bias_loader(value.to(dtype))
            else:
                tp_mod.weight_loader(value.to(dtype))
        else:
            new_param = nn.Parameter(
                value.to(device=device, dtype=dtype),
                requires_grad=False,
            )
            setattr(parent, attr, new_param)

    seen: set[str] = set()
    all_params = list(model.named_parameters(remove_duplicate=False))
    for path, _ in all_params:
        if path not in state_dict:
            raise KeyError(f"missing weight for {path}")
        if path in seen:
            continue
        _materialize(path, state_dict[path])
        seen.add(path)

    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    leftover = set(state_dict) - seen
    if leftover:
        raise RuntimeError(f"unused weights: {sorted(leftover)}")


def _remap_state_dict(
    raw: dict[str, torch.Tensor], config: Qwen3Config
) -> dict[str, torch.Tensor]:
    """Fuse q/k/v → qkv_proj and gate/up → gate_up_proj, same as Llama.
    Qwen3's q_norm / k_norm keys pass through unchanged."""
    remapped: dict[str, torch.Tensor] = {}

    qkv_parts: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    gate_up_parts: dict[tuple[int, str], dict[str, torch.Tensor]] = {}

    for name, tensor in raw.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            continue

        matched = False
        for proj in ("q_proj", "k_proj", "v_proj"):
            key = f"self_attn.{proj}"
            if key in name:
                layer_idx = _extract_layer_idx(name)
                suffix = name.rsplit(".", 1)[-1]
                dict_key = (layer_idx, suffix)
                if dict_key not in qkv_parts:
                    qkv_parts[dict_key] = {}
                qkv_parts[dict_key][proj] = tensor
                matched = True
                break

        if matched:
            continue

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

        remapped[name] = tensor

    for (layer_idx, suffix), parts in qkv_parts.items():
        fused = torch.cat([parts["q_proj"], parts["k_proj"], parts["v_proj"]], dim=0)
        remapped[f"model.layers.{layer_idx}.self_attn.qkv_proj.{suffix}"] = fused

    for (layer_idx, suffix), parts in gate_up_parts.items():
        fused = torch.cat([parts["gate_proj"], parts["up_proj"]], dim=0)
        remapped[f"model.layers.{layer_idx}.mlp.gate_up_proj.{suffix}"] = fused

    return remapped


def _extract_layer_idx(name: str) -> int:
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    raise ValueError(f"Cannot extract layer index from: {name}")

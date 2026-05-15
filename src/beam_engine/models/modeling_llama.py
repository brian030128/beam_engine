"""Inference-only LLaMA model — standalone PyTorch, no vLLM dependencies.

Attention computation is delegated to an ``AttentionContext`` supplied per
forward, so the same model code drives both the paged-attention and
tree-attention baselines.

Supports tensor parallelism via ``beam_engine.distributed``. When the
caller has called ``init_tp(...)`` with ``tp_size > 1``, the QKV / O /
gate_up / down linears become column- and row-parallel; embeddings,
RMSNorms, and the LM head stay replicated on every rank. KV cache
sharding is handled outside the model (the page-driver allocates
``num_kv_heads // tp_size`` heads per rank — see ``page_driver.py``).
"""

from __future__ import annotations

import glob
import os

import flashinfer.activation
import torch
import torch.nn as nn
from transformers import LlamaConfig
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


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], bias=bias
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        return self.down_proj(flashinfer.activation.silu_and_mul(gate_up))


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
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

        # Per-rank head counts. Attention runs on this rank's slice only;
        # the all-reduce inside ``o_proj`` (RowParallelLinear) sums the
        # per-rank slices back into the full hidden_size activation.
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

        # RoPE runs on the per-rank head slice — the kernel just needs the
        # local head counts so it slices q/k along the right axis.
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
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, ctx)
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
        ctx: AttentionContext,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, ctx)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        # lm_head is replicated — small enough that the cost of sharding +
        # all-gather outweighs the savings, and keeping it replicated lets
        # every rank compute the same logits → identical CPU beam state
        # without an extra collective on the hot path.
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
    ) -> "LlamaForCausalLM":
        config = LlamaConfig.from_pretrained(model_name)

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
    """Materialize each parameter on the target device, slicing TP-sharded
    weights through the layer's ``weight_loader``.

    Replicated parameters (embeddings, RMSNorms, lm_head) just get the
    full tensor copied. TP-sharded parameters (anything inside a
    ``Column/Row/QKV/MergedColumnParallelLinear``) route through the
    module's ``weight_loader`` / ``bias_loader``, which slices this
    rank's piece out of the full tensor.

    Using meta-init + per-parameter materialization keeps peak host
    memory at one copy of the model (the CPU state_dict) — same as the
    old ``model.load_state_dict(..., assign=True)`` path.
    """
    tp_layers = (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )

    # Index each TP-aware module by its full state-dict prefix so we can
    # route ``<prefix>.weight`` and ``<prefix>.bias`` keys to the right
    # loader.
    tp_module_prefixes: dict[str, nn.Module] = {}
    for full_name, module in model.named_modules():
        if isinstance(module, tp_layers):
            tp_module_prefixes[full_name] = module

    def _materialize(param_path: str, value: torch.Tensor) -> None:
        # Replace the meta-device parameter with a properly-shaped one on
        # the target device, then copy the value (possibly through a
        # TP-aware loader).
        parent_path, _, attr = param_path.rpartition(".")
        parent: nn.Module = model.get_submodule(parent_path) if parent_path else model
        existing = getattr(parent, attr)
        is_bias = attr == "bias"

        owner_prefix = None
        for prefix in tp_module_prefixes:
            if param_path == f"{prefix}.weight" or param_path == f"{prefix}.bias":
                owner_prefix = prefix
                break

        if owner_prefix is not None:
            tp_mod = tp_module_prefixes[owner_prefix]
            target_param = getattr(tp_mod, attr)
            # Allocate on-device with the per-partition shape, then call
            # the loader to copy our slice in.
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
            # Replicated parameter — full copy.
            new_param = nn.Parameter(
                value.to(device=device, dtype=dtype),
                requires_grad=False,
            )
            setattr(parent, attr, new_param)

    seen: set[str] = set()
    # ``named_parameters()`` dedupes tied parameters — when
    # ``tie_word_embeddings=True`` only ``model.embed_tokens.weight`` is
    # yielded, not ``lm_head.weight``, so we'd leave the lm_head Parameter
    # on meta. Snapshot the list first, then explicitly walk all distinct
    # parameter *paths* (using ``remove_duplicate=False``).
    all_params = list(model.named_parameters(remove_duplicate=False))
    for path, _ in all_params:
        if path not in state_dict:
            raise KeyError(f"missing weight for {path}")
        if path in seen:
            continue
        _materialize(path, state_dict[path])
        seen.add(path)

    # Re-establish weight tying after materialization. Each call to
    # ``_materialize`` swaps in a fresh ``nn.Parameter`` on the owning
    # module, so even though we copied lm_head's values from the same
    # tensor, the two slots now hold independent Parameters. Point them
    # back at the same storage so tied behavior matches the pre-meta
    # construction.
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    leftover = set(state_dict) - seen
    if leftover:
        raise RuntimeError(f"unused weights: {sorted(leftover)}")


def _remap_state_dict(
    raw: dict[str, torch.Tensor], config: LlamaConfig
) -> dict[str, torch.Tensor]:
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

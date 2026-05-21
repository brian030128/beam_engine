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
from ..quantization import (
    Fp8Config,
    detect_quant_config,
    fuse_fp8_gate_up_scales,
    fuse_fp8_qkv_scales,
    is_layer_ignored,
)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        *,
        quant: Fp8Config | None = None,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], bias=bias,
            quant=quant,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=bias, quant=quant,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        return self.down_proj(flashinfer.activation.silu_and_mul(gate_up))


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        *,
        quant: Fp8Config | None = None,
    ):
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
            quant=quant,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads_total * self.head_dim, self.hidden_size, bias=bias,
            quant=quant,
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
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        *,
        quant: Fp8Config | None = None,
    ):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx, quant=quant)
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=getattr(config, "mlp_bias", False),
            quant=quant,
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
    def __init__(
        self,
        config: LlamaConfig,
        *,
        quant: Fp8Config | None = None,
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i, quant=quant)
             for i in range(config.num_hidden_layers)]
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
    def __init__(
        self,
        config: LlamaConfig,
        *,
        quant: Fp8Config | None = None,
    ):
        super().__init__()
        self.config = config
        self.quant = quant
        self.model = LlamaModel(config, quant=quant)
        # lm_head is replicated and stays unquantized: AutoFP8 leaves
        # lm_head in bf16 (it's in ``ignored_layers``), so we keep the
        # standard nn.Linear here. The cost of sharding + all-gather
        # would outweigh the savings, and keeping it replicated lets
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

        model_dir = snapshot_download(
            model_name,
            allow_patterns=["*.safetensors", "*.json"],
        )
        quant = detect_quant_config(model_dir)

        with torch.device("meta"):
            model = cls(config, quant=quant)

        safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        raw_state_dict: dict[str, torch.Tensor] = {}
        for f in safetensor_files:
            raw_state_dict.update(load_file(f, device="cpu"))

        if quant is not None:
            remapped = _remap_state_dict_fp8(raw_state_dict, config)
            if "lm_head.weight" not in remapped and "model.embed_tokens.weight" in remapped:
                remapped["lm_head.weight"] = remapped["model.embed_tokens.weight"]
            _load_into_model_fp8(model, remapped, quant, dtype=dtype, device=device)
        else:
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
    """Standard (unquantized) remap: fuses q/k/v → qkv_proj.weight and
    gate/up → gate_up_proj.weight. For fp8 checkpoints use
    ``_remap_state_dict_fp8`` instead — that one carries the per-tensor
    scales alongside the weights."""
    remapped: dict[str, torch.Tensor] = {}

    qkv_parts: dict[int, dict[str, torch.Tensor]] = {}
    gate_up_parts: dict[int, dict[str, torch.Tensor]] = {}

    for name, tensor in raw.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            continue

        matched = False
        for proj in ("q_proj", "k_proj", "v_proj"):
            key = f"self_attn.{proj}.weight"
            if key in name:
                layer_idx = _extract_layer_idx(name)
                qkv_parts.setdefault(layer_idx, {})[proj] = tensor
                matched = True
                break
        if matched:
            continue

        for proj in ("gate_proj", "up_proj"):
            key = f"mlp.{proj}.weight"
            if key in name:
                layer_idx = _extract_layer_idx(name)
                gate_up_parts.setdefault(layer_idx, {})[proj] = tensor
                matched = True
                break
        if matched:
            continue

        remapped[name] = tensor

    for layer_idx, parts in qkv_parts.items():
        fused = torch.cat(
            [parts["q_proj"], parts["k_proj"], parts["v_proj"]], dim=0,
        )
        remapped[f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"] = fused

    for layer_idx, parts in gate_up_parts.items():
        fused = torch.cat([parts["gate_proj"], parts["up_proj"]], dim=0)
        remapped[f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"] = fused

    return remapped


def _remap_state_dict_fp8(
    raw: dict[str, torch.Tensor], config: LlamaConfig
) -> dict[str, torch.Tensor]:
    """Remap a compressed-tensors / AutoFP8 raw checkpoint.

    Each fp8 linear ships ``weight``, ``weight_scale``, ``input_scale``.
    For q/k/v and gate/up we fuse all three along dim 0, broadcasting
    the per-tensor weight_scale of each chunk to a per-output-channel
    vector before cat (so the fused linear carries a single per-channel
    weight_scale matching the fused weight's row layout). The fused
    input_scale is the max across the chunks — every chunk now shares
    one activation scaling, so we pick the safest value.

    Unquantized weights (norms, embeddings, lm_head) pass through
    untouched. The lm_head is in AutoFP8's ``ignored_layers`` so it
    stays as bf16/fp16 in the raw checkpoint.
    """
    remapped: dict[str, torch.Tensor] = {}

    # Per layer, per suffix ∈ {weight, weight_scale, input_scale}: parts
    # keyed by projection name.
    qkv_parts: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    gate_up_parts: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    qkv_weight_shapes: dict[int, dict[str, tuple[int, ...]]] = {}
    gate_up_weight_shapes: dict[int, dict[str, tuple[int, ...]]] = {}

    for name, tensor in raw.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            continue

        matched = False
        for proj in ("q_proj", "k_proj", "v_proj"):
            key = f"self_attn.{proj}"
            if f".{key}." in f".{name}.":
                layer_idx = _extract_layer_idx(name)
                suffix = name.rsplit(".", 1)[-1]
                qkv_parts.setdefault((layer_idx, suffix), {})[proj] = tensor
                if suffix == "weight":
                    qkv_weight_shapes.setdefault(layer_idx, {})[proj] = tuple(tensor.shape)
                matched = True
                break
        if matched:
            continue

        for proj in ("gate_proj", "up_proj"):
            key = f"mlp.{proj}"
            if f".{key}." in f".{name}.":
                layer_idx = _extract_layer_idx(name)
                suffix = name.rsplit(".", 1)[-1]
                gate_up_parts.setdefault((layer_idx, suffix), {})[proj] = tensor
                if suffix == "weight":
                    gate_up_weight_shapes.setdefault(layer_idx, {})[proj] = tuple(tensor.shape)
                matched = True
                break
        if matched:
            continue

        remapped[name] = tensor

    for (layer_idx, suffix), parts in qkv_parts.items():
        out_prefix = f"model.layers.{layer_idx}.self_attn.qkv_proj"
        if suffix == "weight":
            fused = torch.cat(
                [parts["q_proj"], parts["k_proj"], parts["v_proj"]], dim=0,
            )
            remapped[f"{out_prefix}.weight"] = fused
        elif suffix == "weight_scale":
            shapes = qkv_weight_shapes[layer_idx]
            q_size = shapes["q_proj"][0]
            kv_size = shapes["k_proj"][0]
            # k and v share the same size by construction.
            fused = fuse_fp8_qkv_scales(
                parts["q_proj"], parts["k_proj"], parts["v_proj"],
                q_size=q_size, kv_size=kv_size,
            )
            remapped[f"{out_prefix}.weight_scale"] = fused
        elif suffix == "input_scale":
            # Per-tensor activation scale: take the max across q/k/v so
            # the fused linear is safe under any input.
            fused = torch.stack(
                [parts[p].to(torch.float32).reshape(())
                 for p in ("q_proj", "k_proj", "v_proj")]
            ).max()
            remapped[f"{out_prefix}.input_scale"] = fused

    for (layer_idx, suffix), parts in gate_up_parts.items():
        out_prefix = f"model.layers.{layer_idx}.mlp.gate_up_proj"
        if suffix == "weight":
            fused = torch.cat([parts["gate_proj"], parts["up_proj"]], dim=0)
            remapped[f"{out_prefix}.weight"] = fused
        elif suffix == "weight_scale":
            chunk_size = gate_up_weight_shapes[layer_idx]["gate_proj"][0]
            fused = fuse_fp8_gate_up_scales(
                parts["gate_proj"], parts["up_proj"], chunk_size=chunk_size,
            )
            remapped[f"{out_prefix}.weight_scale"] = fused
        elif suffix == "input_scale":
            fused = torch.stack(
                [parts[p].to(torch.float32).reshape(())
                 for p in ("gate_proj", "up_proj")]
            ).max()
            remapped[f"{out_prefix}.input_scale"] = fused

    return remapped


def _extract_layer_idx(name: str) -> int:
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    raise ValueError(f"Cannot extract layer index from: {name}")


def _load_into_model_fp8(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    quant: Fp8Config,
    *,
    dtype: torch.dtype,
    device: str | torch.device,
) -> None:
    """Materialize an fp8 quantized model.

    Per-parameter dtype rules:
      - fp8 Linear ``.weight``: stays ``float8_e4m3fn``
      - fp8 Linear ``.weight_scale`` / ``.input_scale``: ``float32``
      - everything else (norms, embed, lm_head): compute ``dtype`` (bf16)

    TP-aware loaders (weight_loader / weight_scale_loader /
    input_scale_loader / bias_loader) handle per-rank slicing.
    """
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

    def _owner_prefix(param_path: str) -> str | None:
        for prefix in tp_module_prefixes:
            if param_path.startswith(prefix + "."):
                # The owner is the longest matching prefix; named_modules
                # yields parents first so a Linear that contains no
                # nested ParallelLinear won't be ambiguous here.
                return prefix
        return None

    def _materialize(param_path: str, value: torch.Tensor) -> None:
        parent_path, _, attr = param_path.rpartition(".")
        parent: nn.Module = model.get_submodule(parent_path) if parent_path else model
        existing = getattr(parent, attr)

        owner_prefix = _owner_prefix(param_path)
        if owner_prefix is not None and not is_layer_ignored(param_path, quant.ignored_layers):
            tp_mod = tp_module_prefixes[owner_prefix]
            target_param = getattr(tp_mod, attr)
            # Allocate on-device with the per-partition shape + correct
            # parameter dtype, then call the matching loader.
            new_param = nn.Parameter(
                torch.empty_like(target_param, device=device, dtype=target_param.dtype),
                requires_grad=False,
            )
            setattr(tp_mod, attr, new_param)

            if attr == "weight":
                tp_mod.weight_loader(value)
            elif attr == "bias":
                tp_mod.bias_loader(value.to(dtype))
            elif attr == "weight_scale":
                tp_mod.weight_scale_loader(value)
            elif attr == "input_scale":
                tp_mod.input_scale_loader(value)
            else:
                raise RuntimeError(
                    f"unexpected fp8 TP param attr {attr!r} at {param_path}"
                )
        else:
            # Replicated parameter — full copy. ignored_layers (lm_head)
            # also lands here; its raw safetensors weight is already
            # bf16/fp16, just cast to compute dtype.
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
        raise RuntimeError(f"unused fp8 weights: {sorted(leftover)}")

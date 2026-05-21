"""FP8 loader unit test — pure synthetic fixture, no real model touched.

Constructs a tiny ``raw_state_dict`` matching the AutoFP8 /
compressed-tensors layout (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/
down_proj each with .weight + .weight_scale + .input_scale), runs it
through ``_remap_state_dict_fp8`` + ``_load_into_model_fp8``, and asserts
the resulting model has:

  * fp8 weights on the ParallelLinears
  * float32 per-output-channel weight_scale Parameters with the right shape
  * float32 scalar input_scale Parameters
  * bf16 norms / embeddings / lm_head

Runs on CPU (no GPU required) so this can sit in a pre-step of the
slurm smoke job and catch loader bugs before the 70B torchrun fires.

Does NOT reference any model name string or hit HuggingFace Hub.
"""

from __future__ import annotations

import torch
from transformers import LlamaConfig

from beam_engine.models.modeling_llama import (
    LlamaForCausalLM,
    _load_into_model_fp8,
    _remap_state_dict_fp8,
)
from beam_engine.quantization import Fp8Config


def _make_tiny_fp8_state_dict(config: LlamaConfig) -> dict[str, torch.Tensor]:
    """Build an in-memory raw safetensors-equivalent state_dict mimicking
    the AutoFP8 layout (one fp8 weight + two fp32 scales per linear)."""
    hidden = config.hidden_size
    inter = config.intermediate_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = hidden // num_heads
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    vocab = config.vocab_size

    sd: dict[str, torch.Tensor] = {}
    sd["model.embed_tokens.weight"] = torch.randn(vocab, hidden, dtype=torch.bfloat16)
    sd["model.norm.weight"] = torch.randn(hidden, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(vocab, hidden, dtype=torch.bfloat16)

    def _fp8(shape):
        # Make tensor values stay in a sane range for fp8_e4m3fn
        # (saturates around ±448).
        x = torch.randn(*shape) * 0.5
        return x.clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)

    for li in range(config.num_hidden_layers):
        pfx = f"model.layers.{li}"
        sd[f"{pfx}.input_layernorm.weight"] = torch.randn(hidden, dtype=torch.bfloat16)
        sd[f"{pfx}.post_attention_layernorm.weight"] = torch.randn(hidden, dtype=torch.bfloat16)

        # Attention projections (separate q/k/v/o in raw checkpoint).
        sd[f"{pfx}.self_attn.q_proj.weight"] = _fp8((q_size, hidden))
        sd[f"{pfx}.self_attn.q_proj.weight_scale"] = torch.tensor(0.05, dtype=torch.float32)
        sd[f"{pfx}.self_attn.q_proj.input_scale"] = torch.tensor(0.07, dtype=torch.float32)

        sd[f"{pfx}.self_attn.k_proj.weight"] = _fp8((kv_size, hidden))
        sd[f"{pfx}.self_attn.k_proj.weight_scale"] = torch.tensor(0.04, dtype=torch.float32)
        sd[f"{pfx}.self_attn.k_proj.input_scale"] = torch.tensor(0.08, dtype=torch.float32)

        sd[f"{pfx}.self_attn.v_proj.weight"] = _fp8((kv_size, hidden))
        sd[f"{pfx}.self_attn.v_proj.weight_scale"] = torch.tensor(0.06, dtype=torch.float32)
        sd[f"{pfx}.self_attn.v_proj.input_scale"] = torch.tensor(0.09, dtype=torch.float32)

        sd[f"{pfx}.self_attn.o_proj.weight"] = _fp8((hidden, q_size))
        sd[f"{pfx}.self_attn.o_proj.weight_scale"] = torch.tensor(0.03, dtype=torch.float32)
        sd[f"{pfx}.self_attn.o_proj.input_scale"] = torch.tensor(0.10, dtype=torch.float32)

        # MLP projections (separate gate/up/down in raw checkpoint).
        sd[f"{pfx}.mlp.gate_proj.weight"] = _fp8((inter, hidden))
        sd[f"{pfx}.mlp.gate_proj.weight_scale"] = torch.tensor(0.02, dtype=torch.float32)
        sd[f"{pfx}.mlp.gate_proj.input_scale"] = torch.tensor(0.11, dtype=torch.float32)

        sd[f"{pfx}.mlp.up_proj.weight"] = _fp8((inter, hidden))
        sd[f"{pfx}.mlp.up_proj.weight_scale"] = torch.tensor(0.025, dtype=torch.float32)
        sd[f"{pfx}.mlp.up_proj.input_scale"] = torch.tensor(0.12, dtype=torch.float32)

        sd[f"{pfx}.mlp.down_proj.weight"] = _fp8((hidden, inter))
        sd[f"{pfx}.mlp.down_proj.weight_scale"] = torch.tensor(0.015, dtype=torch.float32)
        sd[f"{pfx}.mlp.down_proj.input_scale"] = torch.tensor(0.13, dtype=torch.float32)
    return sd


def _tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )


def test_remap_fuses_qkv_weight_scale_per_channel():
    """After fusion the per-output-channel weight_scale is a [q+2*kv] vector
    with the right values broadcast over each chunk."""
    cfg = _tiny_llama_config()
    sd = _make_tiny_fp8_state_dict(cfg)
    remapped = _remap_state_dict_fp8(sd, cfg)

    q_size = cfg.num_attention_heads * (cfg.hidden_size // cfg.num_attention_heads)
    kv_size = cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)

    ws = remapped["model.layers.0.self_attn.qkv_proj.weight_scale"]
    assert ws.shape == (q_size + 2 * kv_size,), ws.shape
    assert ws.dtype == torch.float32
    assert torch.allclose(ws[:q_size], torch.full((q_size,), 0.05))
    assert torch.allclose(ws[q_size : q_size + kv_size], torch.full((kv_size,), 0.04))
    assert torch.allclose(ws[q_size + kv_size :], torch.full((kv_size,), 0.06))


def test_remap_qkv_input_scale_is_max():
    cfg = _tiny_llama_config()
    sd = _make_tiny_fp8_state_dict(cfg)
    remapped = _remap_state_dict_fp8(sd, cfg)
    in_s = remapped["model.layers.0.self_attn.qkv_proj.input_scale"]
    # Per fixture: q=0.07, k=0.08, v=0.09 → max = 0.09.
    assert in_s.dtype == torch.float32
    assert in_s.dim() == 0
    assert torch.isclose(in_s, torch.tensor(0.09))


def test_remap_gate_up_fuses_weight_and_scale():
    cfg = _tiny_llama_config()
    sd = _make_tiny_fp8_state_dict(cfg)
    remapped = _remap_state_dict_fp8(sd, cfg)

    w = remapped["model.layers.0.mlp.gate_up_proj.weight"]
    assert w.shape == (2 * cfg.intermediate_size, cfg.hidden_size)
    assert w.dtype == torch.float8_e4m3fn

    ws = remapped["model.layers.0.mlp.gate_up_proj.weight_scale"]
    assert ws.shape == (2 * cfg.intermediate_size,)
    assert torch.allclose(ws[: cfg.intermediate_size], torch.full((cfg.intermediate_size,), 0.02))
    assert torch.allclose(ws[cfg.intermediate_size :], torch.full((cfg.intermediate_size,), 0.025))


def test_load_into_model_fp8_materializes_correct_dtypes():
    """End-to-end: raw checkpoint → fused remap → model. Weights stay fp8,
    scales stay float32, norms/embeddings/lm_head get bf16."""
    cfg = _tiny_llama_config()
    sd = _make_tiny_fp8_state_dict(cfg)
    quant = Fp8Config(
        quant_method="fp8",
        activation_scheme="static",
        ignored_layers=("lm_head",),
    )

    with torch.device("meta"):
        model = LlamaForCausalLM(cfg, quant=quant)

    remapped = _remap_state_dict_fp8(sd, cfg)
    _load_into_model_fp8(model, remapped, quant, dtype=torch.bfloat16, device="cpu")

    # Pick a representative layer 0.
    attn = model.model.layers[0].self_attn
    mlp = model.model.layers[0].mlp

    assert attn.qkv_proj.weight.dtype == torch.float8_e4m3fn
    assert attn.qkv_proj.weight_scale.dtype == torch.float32
    assert attn.qkv_proj.weight_scale.shape == (
        cfg.num_attention_heads * (cfg.hidden_size // cfg.num_attention_heads)
        + 2 * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads),
    )
    assert attn.qkv_proj.input_scale.dtype == torch.float32
    assert attn.qkv_proj.input_scale.dim() == 0

    assert attn.o_proj.weight.dtype == torch.float8_e4m3fn
    # Row-parallel: weight_scale is broadcast to [out_features] (per-channel,
    # replicated value).
    assert attn.o_proj.weight_scale.shape == (cfg.hidden_size,)

    assert mlp.gate_up_proj.weight.dtype == torch.float8_e4m3fn
    assert mlp.gate_up_proj.weight_scale.shape == (2 * cfg.intermediate_size,)

    assert mlp.down_proj.weight.dtype == torch.float8_e4m3fn
    assert mlp.down_proj.weight_scale.shape == (cfg.hidden_size,)

    # Replicated tensors take the compute dtype.
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.model.norm.weight.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.bfloat16


if __name__ == "__main__":
    test_remap_fuses_qkv_weight_scale_per_channel()
    test_remap_qkv_input_scale_is_max()
    test_remap_gate_up_fuses_weight_and_scale()
    test_load_into_model_fp8_materializes_correct_dtypes()
    print("OK")

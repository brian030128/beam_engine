"""FP8 quantization helpers.

Targets pre-quantized checkpoints in AutoFP8 / compressed-tensors format
(e.g. ``RedHatAI/Meta-Llama-3-70B-Instruct-FP8``). Each quantized linear
ships three tensors:

  * ``<prefix>.weight``        — ``float8_e4m3fn``, shape ``[out, in]``
  * ``<prefix>.weight_scale``  — ``float32`` scalar (per-tensor)
  * ``<prefix>.input_scale``   — ``float32`` scalar (static activation scale)

For matmul we use ``torch._scaled_mm`` (FP8 tensor-core path, H100+):

    x_fp8 = (x_bf16 / input_scale).clamp(-FP8_MAX, FP8_MAX).to(float8_e4m3fn)
    out   = _scaled_mm(x_fp8, w_fp8.t(),
                       scale_a=input_scale,
                       scale_b=weight_scale_per_channel,
                       out_dtype=bf16)

Activations use the static ``input_scale`` saved in the checkpoint
(``activation_scheme: static``). Fused linears (qkv, gate_up) carry a
per-output-channel ``weight_scale`` built from the three (or two) original
per-tensor scales — one row per chunk, broadcast to all channels in that
chunk — so a single ``_scaled_mm`` covers the fused output. Their
``input_scale`` is the max across the per-chunk static scales (they share
the same input, so over-quantizing the lower-scaled chunks is the trade).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch


FP8_E4M3_MAX = 448.0


@dataclass(frozen=True)
class Fp8Config:
    """Lightweight view of ``config.json#quantization_config`` for FP8.

    We only care about the bits that affect loading + forward: the quant
    method, the activation scheme (static vs dynamic), and which layer
    prefixes are unquantized (``ignored_layers``, e.g. ``lm_head``).
    """
    quant_method: str = "fp8"
    activation_scheme: str = "static"
    ignored_layers: tuple[str, ...] = field(default_factory=tuple)


def detect_quant_config(model_dir: str) -> Optional[Fp8Config]:
    """Return an ``Fp8Config`` if the checkpoint's ``config.json`` declares
    per-tensor static fp8 quantization; otherwise ``None``. Reads only
    ``config.json``, so this is cheap and side-effect-free.

    Two upstream schemas land here:
    - **AutoFP8** (``quant_method: fp8``, e.g. RedHatAI's 70B-FP8) —
      per-tensor static scales, optional ``activation_scheme: static``.
      Block-quantized variants (``weight_block_size`` set, e.g.
      Qwen/Qwen3-4B-FP8) are rejected; they need a different GEMM path.
    - **compressed-tensors / naive-quantized** (``quant_method:
      compressed-tensors`` + ``format: naive-quantized``, e.g.
      RedHatAI's Llama-3.1-8B-Instruct-FP8) — same per-tensor static fp8
      under a different field schema. Validated by checking each
      ``config_groups.*.weights`` and ``input_activations`` block.
    """
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    qcfg = cfg.get("quantization_config")
    if not qcfg:
        return None
    method = qcfg.get("quant_method", "").lower()

    if method == "fp8":
        # Block-quantized fp8 (Qwen3-FP8, DeepSeek-V3 style) ships a
        # ``weight_block_size`` and uses ``weight_scale_inv`` per-block
        # rather than per-tensor ``weight_scale``. The fp8 GEMM path
        # here only handles per-tensor scales — refuse to claim this
        # checkpoint and let the caller see a load error.
        if qcfg.get("weight_block_size"):
            return None
        return Fp8Config(
            quant_method="fp8",
            activation_scheme=qcfg.get("activation_scheme", "static"),
            ignored_layers=tuple(qcfg.get("ignored_layers", ())),
        )

    if method == "compressed-tensors":
        # naive-quantized = the legacy AutoFP8 layout (Llama-3.1-8B-FP8).
        # float-quantized = newer label for the same per-tensor static
        # fp8 layout (Llama-3.2-1B-FP8). Both share the same on-disk
        # shape (weight + weight_scale + input_scale per linear) so the
        # same remap+load path applies.
        if qcfg.get("format") not in ("naive-quantized", "float-quantized"):
            return None
        groups = qcfg.get("config_groups") or {}
        if not groups:
            return None
        for g in groups.values():
            w = g.get("weights") or {}
            if w.get("num_bits") != 8 or w.get("type") != "float":
                return None
            if w.get("strategy") != "tensor":
                return None
            a = g.get("input_activations") or {}
            if a:
                if a.get("num_bits") != 8 or a.get("type") != "float":
                    return None
                if a.get("strategy") != "tensor" or a.get("dynamic"):
                    return None
        return Fp8Config(
            quant_method="fp8",
            activation_scheme="static",
            ignored_layers=tuple(qcfg.get("ignore", ())),
        )

    # Other quant methods (gptq/awq/compressed-tensors-int8/…) flow to
    # the unquantized path; callers will see a load error from the
    # standard loader and we can extend here when needed.
    return None


def is_layer_ignored(param_path: str, ignored_layers: Sequence[str]) -> bool:
    """Match the rules used by compressed-tensors / AutoFP8: a layer is
    ignored if its full state-dict path contains any of the
    ``ignored_layers`` strings (typically ``"lm_head"``).
    """
    return any(ig in param_path for ig in ignored_layers)


def quantize_to_fp8_e4m3(
    x: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Quantize a real-valued tensor to ``float8_e4m3fn`` using the
    multiplicative scale ``scale`` (i.e. ``x_real ≈ x_fp8 * scale``).

    ``scale`` may be a 0-d tensor or scalar Python float.
    """
    x_real = x.to(torch.float32)
    inv = 1.0 / scale.to(torch.float32) if torch.is_tensor(scale) else 1.0 / scale
    return (x_real * inv).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)


def fp8_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 GEMM equivalent to ``F.linear(x, weight, bias)`` for an FP8
    checkpoint with per-tensor activation scale and per-output-channel
    weight scale.

    ``x``: real-valued activation, shape ``[..., in]`` (any leading dims).
    ``weight_fp8``: ``[out, in]``, ``float8_e4m3fn``.
    ``weight_scale``: ``[out]`` (per-channel) or scalar.
    ``input_scale``: scalar (per-tensor static).

    Uses ``torch._scaled_mm`` in **RowWise** mode (the only mode that
    supports per-tensor activation + per-channel weight on H100): both
    scales are 2-D contiguous (``scale_a`` ``[M, 1]``, ``scale_b``
    ``[1, N]``). We build them per-call from the stored per-tensor /
    per-channel parameters — the (M, 1) allocation is M·4 bytes
    (negligible vs the matmul).
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M = x_2d.shape[0]
    x_fp8 = quantize_to_fp8_e4m3(x_2d, input_scale)

    # _scaled_mm RowWise requires:
    #   scale_a shape (M, 1), contiguous, float32
    #   scale_b shape (1, N), contiguous, float32
    in_scale_f32 = input_scale.to(torch.float32).reshape(())
    scale_a = in_scale_f32.expand(M).contiguous().view(M, 1)
    w_scale_f32 = weight_scale.to(torch.float32)
    if w_scale_f32.dim() == 0:
        # Single-chunk linears stored a 0-d per-tensor scale (we now
        # broadcast to per-channel uniformly at load time, but this
        # branch keeps the helper defensive).
        scale_b = w_scale_f32.expand(weight_fp8.shape[0]).contiguous().view(1, -1)
    else:
        scale_b = w_scale_f32.contiguous().view(1, -1)

    out = torch._scaled_mm(
        x_fp8,
        weight_fp8.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=x.dtype,
    )
    if bias is not None:
        out = out + bias
    return out.view(*orig_shape[:-1], -1)


def fuse_fp8_qkv_scales(
    s_q: torch.Tensor,
    s_k: torch.Tensor,
    s_v: torch.Tensor,
    *,
    q_size: int,
    kv_size: int,
) -> torch.Tensor:
    """Broadcast per-tensor scales for q/k/v into a per-output-channel
    ``[q_size + 2*kv_size]`` scale vector matching the fused
    ``QKVParallelLinear`` weight's row layout.
    """
    parts = [
        s_q.to(torch.float32).expand(q_size).contiguous(),
        s_k.to(torch.float32).expand(kv_size).contiguous(),
        s_v.to(torch.float32).expand(kv_size).contiguous(),
    ]
    return torch.cat(parts, dim=0)


def fuse_fp8_gate_up_scales(
    s_gate: torch.Tensor,
    s_up: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Per-output-channel scale for the fused ``gate_up_proj`` weight."""
    parts = [
        s_gate.to(torch.float32).expand(chunk_size).contiguous(),
        s_up.to(torch.float32).expand(chunk_size).contiguous(),
    ]
    return torch.cat(parts, dim=0)

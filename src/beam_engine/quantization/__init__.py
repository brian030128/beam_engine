"""Quantization support for beam_engine.

Currently only FP8 (compressed-tensors / AutoFP8 format) is implemented;
see ``fp8.py``. The detection entry point ``detect_quant_config`` returns
``None`` for unquantized checkpoints, which lets ``modeling_llama`` keep
its standard bf16 load path unchanged.
"""

from __future__ import annotations

from .fp8 import (
    Fp8Config,
    detect_quant_config,
    fp8_linear,
    fuse_fp8_qkv_scales,
    fuse_fp8_gate_up_scales,
    is_layer_ignored,
    quantize_to_fp8_e4m3,
)

__all__ = [
    "Fp8Config",
    "detect_quant_config",
    "fp8_linear",
    "fuse_fp8_qkv_scales",
    "fuse_fp8_gate_up_scales",
    "is_layer_ignored",
    "quantize_to_fp8_e4m3",
]

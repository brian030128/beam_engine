"""Model dispatcher — pick the right backbone class based on model name."""

from __future__ import annotations

from typing import Any

import torch


def load_model_for_causal_lm(
    model_name: str,
    *,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cuda",
) -> Any:
    """Load a causal-LM model by name, dispatching to the matching backbone.

    Detection is purely lexical on the HF model id — Qwen3-family names
    contain ``qwen3``, everything else falls through to LlamaForCausalLM.
    Both backbones expose a ``from_pretrained(model_name, dtype, device)``
    classmethod with the same signature.
    """
    name_lower = model_name.lower()
    if "qwen3" in name_lower:
        from .modeling_qwen3 import Qwen3ForCausalLM
        return Qwen3ForCausalLM.from_pretrained(
            model_name, dtype=dtype, device=device,
        )
    from .modeling_llama import LlamaForCausalLM
    return LlamaForCausalLM.from_pretrained(
        model_name, dtype=dtype, device=device,
    )

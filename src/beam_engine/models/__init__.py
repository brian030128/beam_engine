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
    """Load a causal-LM model by name.

    The paper evaluates the Llama-3 family (1B / 8B / 70B), all served by
    ``LlamaForCausalLM`` via ``from_pretrained(model_name, dtype, device)``.
    """
    from .modeling_llama import LlamaForCausalLM
    return LlamaForCausalLM.from_pretrained(
        model_name, dtype=dtype, device=device,
    )

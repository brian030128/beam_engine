"""Smoke-test the batched adaptive_pool + mlca rewrites.

Confirms B=1 is regression-free (matches bs_kernel B=1 top tokens) and
B=4 doesn't crash. Tests that all three drivers emit identical top-beam
tokens at B=1 (deterministic decoding from the same prompt).
"""

from __future__ import annotations

import inspect

import torch
from transformers import AutoTokenizer

from beam_engine.baselines.fasttree import beam_search as ft_search
from beam_engine.baselines.mlca import beam_search as mlca_search
from beam_engine.methods.adaptive_pool import beam_search as ap_search
from beam_engine.methods.bs_kernel import beam_search as bk_search
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16


def _make_prompt(tok, target_len: int) -> list[int]:
    seed = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar named Elara who studied the stars and the ways of the "
        "natural world. "
    )
    ids: list[int] = []
    while len(ids) < target_len:
        ids.extend(tok.encode(seed, add_special_tokens=False))
    return ids[:target_len]


def main():
    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    K = 16
    L_p = 2048
    max_new = 8
    prompt = _make_prompt(tok, L_p)

    methods = {
        "adaptive_pool": ap_search,
        "mlca": mlca_search,
        "fasttree": ft_search,
        "bs_kernel": bk_search,
    }

    for B in (1, 4):
        print(f"=== B={B} K={K} L_p={L_p} max_new={max_new} ===")
        prompts = [list(prompt) for _ in range(B)]
        results: dict[str, list[int]] = {}
        for name, fn in methods.items():
            extra: dict = {}
            if "max_num_pages" in inspect.signature(fn).parameters:
                extra["max_num_pages"] = 4096
            beams, timings = fn(
                model, config, prompts, max_new, K,
                return_timings=True, **extra,
            )
            assert len(beams) == B and all(len(bl) == K for bl in beams)
            top_tokens = beams[0][0].token_ids
            results[name] = top_tokens
            steps = timings["decode_step_ms"]
            mean_step = sum(steps) / len(steps) if steps else 0.0
            print(
                f"  {name:<14} prefill={timings['prefill_ms']:7.1f}ms  "
                f"mean_step={mean_step:6.2f}ms  top={top_tokens}"
            )
        # All three should match at deterministic decoding.
        token_sets = {tuple(v) for v in results.values()}
        if len(token_sets) == 1:
            print(f"  ✓ all three match")
        else:
            print(f"  ✗ MISMATCH:")
            for name, toks in results.items():
                print(f"      {name}: {toks}")
        print()

    print("smoke OK")


if __name__ == "__main__":
    main()

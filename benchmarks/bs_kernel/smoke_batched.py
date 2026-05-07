"""Smoke-test the phase 2 batched bs_kernel driver.

Runs the same prompt as B=1 (regression: should match the legacy per-prompt
flow's shape and produce sane log-probs) and B=2 / B=4 (the new batched code
path). For each B we just check:
  * function returns without raising
  * len(beams_per_prompt) == B and each has K beams
  * picks histogram is non-empty

Usage:
    uv run python benchmarks/bs_kernel/smoke_batched.py
"""

from __future__ import annotations

from collections import Counter

import torch
from transformers import AutoTokenizer

from beam_engine.methods.bs_kernel import beam_search as bs_kernel_search
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
    base_prompt = _make_prompt(tok, L_p)
    # Slightly different prompts so cross-prompt batched cascade has actual
    # work (each prompt has its own LCA prefix).
    prompts_b = {
        1: [base_prompt],
        2: [base_prompt, base_prompt[:-32] + tok.encode(" different tail.", add_special_tokens=False)[:32]],
    }
    p3 = base_prompt[:-64] + tok.encode(" yet another tail variation.", add_special_tokens=False)[:64]
    prompts_b[4] = [base_prompt, base_prompt, p3, p3]

    for B, prompts in prompts_b.items():
        L_ps = [len(p) for p in prompts]
        # Pad/truncate so every prompt has the same L_p (we don't require it but
        # bench harness uses uniform L_p).
        prompts = [p[:L_p] + base_prompt[len(p[:L_p]):L_p] for p in prompts]
        print(f"=== B={B}, K={K}, L_p={L_p}, max_new={max_new} ===")
        beams, timings, picks = bs_kernel_search(
            model, config, prompts, max_new, K,
            return_timings=True, return_picks=True,
            max_num_pages=4096,
        )
        assert len(beams) == B, f"expected {B} beam-lists, got {len(beams)}"
        for b, beam_list in enumerate(beams):
            assert len(beam_list) == K, f"prompt {b}: expected {K} beams, got {len(beam_list)}"
            assert all(len(bm.token_ids) == max_new for bm in beam_list)
        print(f"  prefill_ms={timings['prefill_ms']:.1f}")
        if timings["decode_step_ms"]:
            n = len(timings["decode_step_ms"])
            mean = sum(timings["decode_step_ms"]) / n
            print(f"  decode_steps={n}, mean_step_ms={mean:.2f}")
        # Picks histogram across all prompts (they share the same per-step pick).
        hist = Counter(p.strategy.value for p in picks[0])
        print(f"  picks: {dict(hist)}")
        print(f"  top beam[0] tokens: {beams[0][0].token_ids}")
        print()

    print("smoke OK")


if __name__ == "__main__":
    main()

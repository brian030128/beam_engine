"""Correctness regression test for the bs_kernel decomp cache.

Compares bs_kernel.beam_search (with cache) against paged.beam_search
on a battery of (K, L_p, max_new) cases. Both share the model code and
the paged-prefill phase; the cache only affects bs_kernel's decode
plan_decode_step, so identical token sequences within fp16 tolerance
imply the cache is invalidating correctly.

Why not tests/test_bs_kernel.py? — that compares bs_kernel vs tree.py
which uses flashinfer's SM90 ragged-prefill kernel; on this cluster
that kernel returns ``operation not supported`` regardless of any
bs_kernel changes (see prior log
``slurm/logs/test_bs_kernel_196921.err`` from before this commit).

Avoiding the ragged-prefill code path keeps this an actual regression
test of the cache, not of unrelated env issues.
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import paged
from beam_engine.methods import bs_kernel
from beam_engine.methods.bs_kernel.cost_model import Strategy
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = os.environ.get("BE_TEST_MODEL", "meta-llama/Llama-3.2-1B")
DEVICE = "cuda"
DTYPE = torch.float16

CASES = [
    # (K, L_p, max_new, forced_strategy)
    # K=1 — degenerate, cache hits trivially.
    (1, 128, 16, None),
    # PER_BEAM regime — no cascade, smaller prompt, exercises the
    # bs_kernel cache through a different dispatch.
    (4, 128, 16, None),
    (16, 128, 16, None),
    # SHARED regime — long prompt, multiple decode steps, the cache
    # should hit on most decode steps (page-boundary misses every 16).
    (4, 4096, 32, None),
    (16, 4096, 32, None),
    # Long rollout — exercises page-boundary cache invalidation
    # multiple times.
    (4, 1024, 64, None),
    (16, 1024, 64, None),
    # Forced SHARED — Llama-3.2-1B's smaller heads shift the picker's
    # breakeven so the natural pick stays PER_BEAM at these shapes;
    # force SHARED_2L_1POOL to exercise the cascade path's cache too.
    (4, 4096, 32, "SHARED_2L_1POOL"),
    (16, 4096, 32, "SHARED_2L_1POOL"),
    # Also DEC_TAIL — uses a different dispatch (prefix prefill +
    # per-beam decode + merge); cache feeds the same level decomp.
    (4, 4096, 32, "SHARED_2L_DEC_TAIL"),
    (16, 4096, 32, "SHARED_2L_DEC_TAIL"),
]


def _make_prompt(tokenizer, target_len: int) -> list[int]:
    text = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
    ) * 200
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:target_len]


def _ok(label: str, condition: bool) -> bool:
    print(f"  {label}: {'PASS' if condition else 'FAIL'}")
    return condition


def main():
    print(f"Model: {MODEL_NAME}")
    print("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    all_pass = True

    for (K, L_p, max_new, forced) in CASES:
        prompt_ids = _make_prompt(tok, L_p)
        forced_label = forced if forced else "auto"
        label = f"K={K} L_p={L_p} max_new={max_new} strategy={forced_label}"
        print("=" * 60)
        print(f"Case: {label}")
        print("=" * 60)

        ref_beams = paged.beam_search(
            model, config, [prompt_ids], max_new, K,
        )[0]
        bs_kernel_kwargs = {"return_picks": True}
        if forced is not None:
            bs_kernel_kwargs["available_strategies"] = {Strategy[forced]}
        our_beams, picks_list = bs_kernel.beam_search(
            model, config, [prompt_ids], max_new, K,
            **bs_kernel_kwargs,
        )
        our_beams = our_beams[0]
        picks = picks_list[0]
        pick_counts = dict(Counter(p.strategy.value for p in picks))
        print(f"  picks: {pick_counts}")

        # Token equality, best beam.
        tokens_match = our_beams[0].token_ids == ref_beams[0].token_ids
        all_pass &= _ok(f"best-beam tokens identical", tokens_match)
        if not tokens_match:
            # Show first divergence point for debugging.
            for i, (a, b) in enumerate(
                zip(our_beams[0].token_ids, ref_beams[0].token_ids)
            ):
                if a != b:
                    print(f"    first divergence at step {i}: "
                          f"bs_kernel={a}, paged={b}")
                    break

        score_close = math.isclose(
            ref_beams[0].cum_log_prob, our_beams[0].cum_log_prob,
            rel_tol=1e-2, abs_tol=5e-2,
        )
        all_pass &= _ok(
            f"best-beam scores within 5e-2 "
            f"(ref={ref_beams[0].cum_log_prob:.4f} "
            f"ours={our_beams[0].cum_log_prob:.4f})",
            score_close,
        )

        # All K beams token-set equal (allow rank swaps for low-ranked
        # beams — paged vs SHARED differ by fp16 attention precision so
        # lower-ranked beams' scores can flip ranking by < 1e-4).
        ref_tokens = sorted(tuple(b.token_ids) for b in ref_beams)
        our_tokens = sorted(tuple(b.token_ids) for b in our_beams)
        all_pass &= _ok(
            f"all {K} beams' token sets identical (rank-agnostic)",
            ref_tokens == our_tokens,
        )
        print()

    print("=" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

"""bs_kernel correctness vs tree.py.

`tree.py` is the validated numerical reference (`tests/test_baselines.py`
already establishes that paged.py and tree.py produce identical beam token
sets within fp16 attention drift). This script asserts that bs_kernel.beam_search
matches tree.py across:

  * K ∈ {1, 4, 16}        K=1 verifies the degenerate-to-PER_BEAM fallback
                          (cost model picks PER_BEAM, driver dispatches to
                          BatchDecodeWithPagedKVCacheWrapper).
  * L_p ∈ {128, 4096}     Short prompt → cost model picks PER_BEAM; long
                          prompt → SHARED dispatch through the fused-cascade
                          wrapper. Both branches need to produce the same
                          beams.
  * max_new ∈ {16, 64}    Short and longer rollout.

The full design-doc grid extends to K=64 / max_new=128 — those are perf
test cases gated on memory/runtime, not correctness, and live with the
sweep harness (Step 6).

GPU pinning is the user's responsibility per CLAUDE.md — set
CUDA_VISIBLE_DEVICES=<idle-id> before running.
"""

from __future__ import annotations

import math
import sys
from collections import Counter

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import tree
from beam_engine.methods import bs_kernel
from beam_engine.methods.bs_kernel.cost_model import Strategy
from beam_engine.models.modeling_llama import LlamaForCausalLM


import os
MODEL_NAME = os.environ.get("BE_TEST_MODEL", "meta-llama/Llama-3.1-8B")
DEVICE = "cuda"
DTYPE = torch.float16

# (K, L_p, max_new). Hits both PER_BEAM (short L_p) and SHARED (long L_p)
# dispatches; K=1 case explicitly exercises the degenerate fallback.
CASES = [
    (1, 128, 16),     # greedy fallback
    (4, 128, 16),     # PER_BEAM, basic
    (4, 128, 64),     # PER_BEAM, longer rollout
    (4, 4096, 16),    # SHARED at K=4 (breakeven ≈ 1139 tokens at default coeffs)
    (16, 128, 16),    # PER_BEAM, more beams
    (16, 4096, 16),   # SHARED at K=16 (breakeven ≈ 228 tokens)
]


def _ok(label: str, condition: bool) -> bool:
    print(f"  {label}: {'PASS' if condition else 'FAIL'}")
    return condition


def _make_prompt(tokenizer, target_len: int) -> list[int]:
    """Construct a prompt that tokenizes to exactly target_len tokens."""
    base = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
        "She traveled across mountains and rivers, recording her findings "
        "in a leather-bound journal that grew thicker with every passing "
        "season. The townsfolk welcomed her with warm hearts and cool drinks. "
    ) * 200
    ids = tokenizer.encode(base, add_special_tokens=False)
    if len(ids) < target_len:
        raise RuntimeError(f"base text too short: {len(ids)} < {target_len}")
    return ids[:target_len]


def _compare(label: str, ref_beams, our_beams, *, score_tol: float = 5e-2) -> bool:
    """Beam token-set equality + per-beam score drift within tol."""
    ref_tokens = sorted(tuple(b.token_ids) for b in ref_beams)
    our_tokens = sorted(tuple(b.token_ids) for b in our_beams)
    if ref_tokens != our_tokens:
        # Show the first diverging beam.
        for r, o in zip(ref_tokens, our_tokens):
            if r != o:
                print(f"    {label}: token sets differ")
                print(f"      ref:  {r}")
                print(f"      ours: {o}")
                break
        else:
            print(f"    {label}: token-set length differs ({len(ref_tokens)} vs {len(our_tokens)})")
        return False
    ref_sorted = sorted(ref_beams, key=lambda b: b.token_ids)
    our_sorted = sorted(our_beams, key=lambda b: b.token_ids)
    for rb, ob in zip(ref_sorted, our_sorted):
        if not math.isclose(
            rb.cum_log_prob, ob.cum_log_prob,
            rel_tol=1e-2, abs_tol=score_tol,
        ):
            print(
                f"    {label}: per-beam score drift {abs(rb.cum_log_prob - ob.cum_log_prob):.4f}"
                f" (ref={rb.cum_log_prob:.4f} ours={ob.cum_log_prob:.4f})"
            )
            return False
    return True


def main():
    print("Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    all_pass = True

    for (K, L_p, max_new) in CASES:
        prompt_ids = _make_prompt(tokenizer, L_p)
        label = f"K={K} L_p={L_p} max_new={max_new}"
        print("=" * 60)
        print(f"Case: {label}")
        print("=" * 60)

        ref_beams = tree.beam_search(
            model, config, [prompt_ids], max_new, K,
        )[0]
        our_beams_list, picks_list = bs_kernel.beam_search(
            model, config, [prompt_ids], max_new, K,
            return_picks=True,
        )
        our_beams = our_beams_list[0]
        picks = picks_list[0]
        pick_counts = dict(Counter(p.strategy.value for p in picks))
        print(f"  picks: {pick_counts}")

        # K=1 must always pick PER_BEAM (sharing across one beam is meaningless).
        if K == 1:
            all_pass &= _ok(
                "K=1 always picks PER_BEAM",
                all(p.strategy == Strategy.PER_BEAM for p in picks),
            )

        # L_p ≥ 4096 with K ≥ 4 should pick SHARED (cost model breakeven
        # at default coeffs ≈ 1139 tokens at K=4, ≈ 228 at K=16).
        if L_p >= 4096 and K >= 4:
            all_pass &= _ok(
                "L_p=4096 picks SHARED at least once",
                any(p.strategy != Strategy.PER_BEAM for p in picks),
            )

        all_pass &= _ok(f"matches tree.py ({label})", _compare(label, ref_beams, our_beams))

        # Best-beam score sanity (tighter tolerance than per-beam drift).
        all_pass &= _ok(
            f"best-beam score within 1e-2 ({label})",
            math.isclose(
                ref_beams[0].cum_log_prob, our_beams[0].cum_log_prob,
                rel_tol=1e-2, abs_tol=1e-2,
            ),
        )
        print()

    print("=" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

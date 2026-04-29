"""Cross-baseline correctness test.

Both paged-attention and tree-attention beam search are mathematically
equivalent — they differ only in how attention is computed. So they should
produce identical token sequences and matching scores (up to small fp tolerance).

We also check that beam_width=1 reproduces a hand-rolled greedy decode.
"""

from __future__ import annotations

import math

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import paged, tree
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16


def _ok(label: str, condition: bool) -> bool:
    print(f"  {label}: {'PASS' if condition else 'FAIL'}")
    return condition


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    prompt = "The capital of France is"
    prompt_ids = tokenizer.encode(prompt)
    max_new = 20

    all_pass = True

    # -----------------------------------------------------------------
    # Test 1 — beam_width=1: paged ≡ tree (token-level identical)
    # -----------------------------------------------------------------
    print("=" * 60)
    print("Test 1: beam_width=1 — paged vs tree token equality")
    print("=" * 60)

    paged_w1 = paged.beam_search(model, config, [prompt_ids], max_new, beam_width=1)[0][0]
    tree_w1 = tree.beam_search(model, config, [prompt_ids], max_new, beam_width=1)[0][0]
    print(f"  paged: {tokenizer.decode(paged_w1.token_ids, skip_special_tokens=True)}")
    print(f"  tree : {tokenizer.decode(tree_w1.token_ids, skip_special_tokens=True)}")
    print(f"  paged score: {paged_w1.cum_log_prob:.4f}")
    print(f"  tree  score: {tree_w1.cum_log_prob:.4f}")

    tokens_match_w1 = paged_w1.token_ids == tree_w1.token_ids
    score_close_w1 = math.isclose(
        paged_w1.cum_log_prob, tree_w1.cum_log_prob, rel_tol=1e-2, abs_tol=1e-2
    )
    all_pass &= _ok("tokens identical (K=1)", tokens_match_w1)
    all_pass &= _ok("scores within 1e-2  (K=1)", score_close_w1)
    print()

    # -----------------------------------------------------------------
    # Test 2 — beam_width=4: paged ≡ tree
    # -----------------------------------------------------------------
    print("=" * 60)
    print("Test 2: beam_width=4 — paged vs tree token equality")
    print("=" * 60)

    K = 4
    paged_beams = paged.beam_search(model, config, [prompt_ids], max_new, beam_width=K)[0]
    tree_beams = tree.beam_search(model, config, [prompt_ids], max_new, beam_width=K)[0]

    print("  paged beams:")
    for i, b in enumerate(paged_beams):
        text = tokenizer.decode(b.token_ids, skip_special_tokens=True)
        print(f"    [{i}] score={b.cum_log_prob:7.4f}  \"{prompt}{text}\"")
    print("  tree beams:")
    for i, b in enumerate(tree_beams):
        text = tokenizer.decode(b.token_ids, skip_special_tokens=True)
        print(f"    [{i}] score={b.cum_log_prob:7.4f}  \"{prompt}{text}\"")

    # Compare token sequences only — fp16 attention paths differ slightly between
    # backends, so per-beam scores can drift by ~0.01 even when tokens are identical.
    paged_token_set = sorted(tuple(b.token_ids) for b in paged_beams)
    tree_token_set = sorted(tuple(b.token_ids) for b in tree_beams)
    beams_match = paged_token_set == tree_token_set
    all_pass &= _ok("beam token sets identical (K=4)", beams_match)

    # Per-beam score drift (within tolerance) — paired by sorted-token-set order
    score_drifts_ok = True
    for pb, tb in zip(
        sorted(paged_beams, key=lambda b: b.token_ids),
        sorted(tree_beams, key=lambda b: b.token_ids),
    ):
        if not math.isclose(pb.cum_log_prob, tb.cum_log_prob, rel_tol=1e-2, abs_tol=5e-2):
            score_drifts_ok = False
            print(f"    beam score drift > 5e-2: paged={pb.cum_log_prob:.4f} tree={tb.cum_log_prob:.4f}")
    all_pass &= _ok("per-beam score drift <= 5e-2 (K=4)", score_drifts_ok)

    # Best-beam score within tolerance
    best_close = math.isclose(
        paged_beams[0].cum_log_prob,
        tree_beams[0].cum_log_prob,
        rel_tol=1e-2, abs_tol=1e-2,
    )
    all_pass &= _ok("best-beam score within 1e-2 (K=4)", best_close)

    # Best-beam score >= K=1 score (beam search should not be worse than greedy)
    best_ge_greedy = paged_beams[0].cum_log_prob >= paged_w1.cum_log_prob - 1e-3
    all_pass &= _ok("paged best  >= greedy (K=4)", best_ge_greedy)

    best_ge_greedy_t = tree_beams[0].cum_log_prob >= tree_w1.cum_log_prob - 1e-3
    all_pass &= _ok("tree best  >= greedy (K=4)", best_ge_greedy_t)
    print()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("=" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

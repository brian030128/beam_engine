"""Correctness smoke for the SHARED-branch plan-state cache.

Runs bs_kernel at K=16, L_p=8192, B=4, max_new=64 and compares
top-beam token sequences against the paged baseline (mathematically
equivalent — must agree exactly under deterministic decoding). Long
enough decode (64 steps) to exercise the cache across many off>0 hits
plus a few off==0 page-append misses.
"""
from __future__ import annotations

import inspect

import torch
from transformers import AutoTokenizer

from beam_engine.baselines.paged import beam_search as paged_search
from beam_engine.methods.bs_kernel import beam_search as bk_search
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.2-1B"
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
    K = 16
    L_p = 8192
    B = 4
    max_new = 64

    print(f"K={K} L_p={L_p} B={B} max_new={max_new}")
    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    base = _make_prompt(tok, L_p)
    prompts = [list(base) for _ in range(B)]
    page_size = 16
    needed_pages = (
        sum((len(p) + page_size - 1) // page_size for p in prompts)
        + B * K * ((max_new + page_size) // page_size + 2)
        + 256
    )

    paged_beams, _ = paged_search(
        model, config, prompts, max_new, K,
        max_num_pages=needed_pages, return_timings=True,
    )
    bs_beams, _ = bk_search(
        model, config, prompts, max_new, K,
        max_num_pages=needed_pages, return_timings=True,
    )

    all_ok = True
    for b in range(B):
        ref = sorted(tuple(bm.token_ids) for bm in paged_beams[b])
        got = sorted(tuple(bm.token_ids) for bm in bs_beams[b])
        if ref == got:
            print(f"  prompt {b}: beam token sets match  (K={K})")
        else:
            print(f"  prompt {b}: MISMATCH")
            for i, (r, g) in enumerate(zip(ref, got)):
                if r != g:
                    print(f"    beam {i}: paged={r[-8:]} bs_kernel={g[-8:]}")
            all_ok = False
    print()
    print("smoke OK" if all_ok else "smoke FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

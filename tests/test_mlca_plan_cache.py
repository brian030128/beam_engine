"""Correctness check for MLCA plan-state cache.

Runs MLCA twice on the same prompts:
  - cache enabled (default, ``MLCA_DISABLE_PLAN_CACHE`` unset)
  - cache disabled (``MLCA_DISABLE_PLAN_CACHE=1``)

Then runs paged-attention as a reference. All three must produce identical
token sequences and matching scores. Beam width = 4, max_new = 40.

Run on H100:
  uv run python tests/test_mlca_plan_cache.py
"""

from __future__ import annotations

import functools
import os
import sys

# Force line-buffered stdout so progress + PASS/FAIL lines appear in the
# tee'd SLURM log even if the script exits mid-loop.
print = functools.partial(print, flush=True)

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import mlca, paged
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda"
DTYPE = torch.float16


def _run_mlca(model, config, prompt_ids, max_new, K, *, disable_cache):
    # Toggle the cache by monkey-patching the backend before invocation.
    # We can't pass it through page_driver.beam_search so we patch the
    # MlcaBackend instance via a wrapper.
    import beam_engine.baselines.mlca as mlca_mod
    orig_plan = mlca_mod.MlcaBackend.plan_decode_step

    if disable_cache:
        def no_cache_plan(self, **kwargs):
            self._plan_cache = {}  # clear before AND after every call
            try:
                return orig_plan(self, **kwargs)
            finally:
                self._plan_cache = {}
        mlca_mod.MlcaBackend.plan_decode_step = no_cache_plan
    try:
        return mlca.beam_search(
            model, config, prompt_ids, max_new_tokens=max_new,
            beam_width=K, dtype=DTYPE, device=DEVICE,
        )
    finally:
        mlca_mod.MlcaBackend.plan_decode_step = orig_plan


def main():
    print("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config

    # Use prompts whose tokenized length spans multiple pages
    # (page_size = 16). page_driver's split-page CoW has a pre-existing
    # edge case when L_p < page_size AND not page-aligned, unrelated to
    # the plan cache. Long prompts avoid it and mirror the paper-exp
    # workloads.
    base = (
        "In the late nineteenth century, scientists began to study the "
        "behavior of light in increasingly sophisticated ways. Maxwell's "
        "equations had unified electricity and magnetism into a single "
        "framework, and the speed of light emerged from that framework "
        "as a fundamental constant. By the early twentieth century, the "
        "photoelectric effect and blackbody radiation forced physicists "
        "to confront the quantum nature of light, leading directly to "
        "Einstein's 1905 paper. "
    )
    prompts = [base + "What followed was",
               base + "The implication for chemistry was",
               base + "In the years that followed,"]
    prompt_ids = [tok.encode(p) for p in prompts]
    print(f"  prompt token counts: {[len(p) for p in prompt_ids]}")
    K = 4
    max_new = 40

    print("Running MLCA...")
    out_mlca = mlca.beam_search(
        model, config, prompt_ids, max_new_tokens=max_new,
        beam_width=K, dtype=DTYPE, device=DEVICE,
    )
    print("Running paged-attention reference...")
    out_paged = paged.beam_search(
        model, config, prompt_ids, max_new_tokens=max_new,
        beam_width=K, dtype=DTYPE, device=DEVICE,
    )

    all_pass = True
    for b, (mlca_b, paged_b) in enumerate(zip(out_mlca, out_paged)):
        for k in range(K):
            t_m = mlca_b[k].token_ids
            t_p = paged_b[k].token_ids
            ok = t_m == t_p
            print(f"  prompt={b} beam={k}: mlca==paged: "
                  f"{'PASS' if ok else 'FAIL'}")
            if not ok:
                print(f"    mlca:  {t_m}")
                print(f"    paged: {t_p}")
            all_pass = all_pass and ok

    print()
    print("OVERALL:", "PASS" if all_pass else "FAIL")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

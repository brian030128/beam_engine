"""A/B test: multi_chain_reasoning through tree_batch_decode vs
page_driver.beam_search.

Both paths run bs_kernel.BsKernelBackend on the same B=32, K=4,
L_p=4176, max_new=256 shape. The tree_batch_decode path uses
fork_at_prefill mode (which I claim is semantically identical to
beam_search at this shape). If timings diverge significantly, the
slowdown is an integration cost in tree_driver; if they match, the
slowdown is inherent to bs_kernel at this geometry.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time

import torch
from transformers import AutoTokenizer

from beam_engine.methods import bs_kernel as bs_kernel_mod
from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
from beam_engine.methods.bs_kernel.driver import BsKernelBackend
from beam_engine.models.modeling_llama import LlamaForCausalLM
from beam_engine.tree_driver import tree_batch_decode

sys.path.insert(0, os.path.dirname(__file__))
from sglang_workloads import SCENARIO_BUILDERS


MODEL_NAME = os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")
DEVICE = "cuda"
DTYPE = torch.float16


def _summary(xs):
    if not xs:
        return 0, 0, 0, 0
    s = sorted(xs)
    return sum(xs)/len(xs), statistics.median(xs), s[int(0.99*(len(s)-1))], sum(xs)


def _phase_table(name, t):
    decode_steps = t["decode_step_ms"]
    d_total = sum(decode_steps)
    print(f"\n=== {name} ===")
    print(f"  decode_total={d_total:.1f}ms over {len(decode_steps)} steps")
    print(f"  {'phase':<14} {'mean':>8} {'p50':>8} {'p99':>8} {'total':>10}  {'% decode':>9}")
    # tree_driver names: alloc/plan/forward/topk;
    # page_driver names: cow/plan/forward/topk/fork.
    phases = []
    for cand in ("alloc_ms", "cow_ms"):
        if cand in t: phases.append(cand)
    for p in ("plan_ms", "forward_ms", "topk_ms", "fork_ms"):
        if p in t: phases.append(p)
    for phase in phases:
        xs = t.get(phase, [])
        mean, p50, p99, total = _summary(xs)
        pct = total/d_total*100 if d_total else 0
        print(f"  {phase:<14} {mean:>8.3f} {p50:>8.3f} {p99:>8.3f} {total:>10.2f}  {pct:>8.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config

    spec = SCENARIO_BUILDERS["multi_chain_reasoning"](tok)
    spec.validate()
    B, K = spec.B, spec.K
    print(f"B={B} K={K} max_new={args.max_new}")
    print(f"shared_prefix len (tokens) per prompt: "
          f"{[len(g.shared_prefix_ids) for g in spec.groups[:3]]}...")

    # ---------------- tree_batch_decode (fork_at_prefill) ----------------
    gc.collect(); torch.cuda.empty_cache()
    coeff = load_or_defaults(torch.device(DEVICE))
    backend = BsKernelBackend(coefficients=coeff)
    # Warmup.
    _ = tree_batch_decode(model, config, spec, args.max_new, backend=backend,
                          return_timings=True)
    del backend; gc.collect(); torch.cuda.empty_cache()

    backend = BsKernelBackend(coefficients=coeff)
    t0 = time.perf_counter()
    res_t = tree_batch_decode(model, config, spec, args.max_new, backend=backend,
                              return_timings=True, return_phase_timings=True)
    wall_t = time.perf_counter() - t0
    print(f"\n[tree_batch_decode] wall {wall_t*1000:.1f}ms")
    _phase_table("tree_driver bs_kernel", res_t.timings)
    del backend, res_t; gc.collect(); torch.cuda.empty_cache()

    # ---------------- page_driver.beam_search (canonical path) ----------------
    # Same shape: prompt_ids = [shared_prefix_b for each group], K beams.
    prompts = [g.shared_prefix_ids for g in spec.groups]
    # Warmup.
    _ = bs_kernel_mod.beam_search(
        model, config, prompts, args.max_new, K, return_timings=True,
    )
    gc.collect(); torch.cuda.empty_cache()

    t0 = time.perf_counter()
    _beams, tim_p = bs_kernel_mod.beam_search(
        model, config, prompts, args.max_new, K,
        return_timings=True, return_phase_timings=True,
    )
    wall_p = time.perf_counter() - t0
    print(f"\n[page_driver.beam_search] wall {wall_p*1000:.1f}ms")
    _phase_table("page_driver bs_kernel", tim_p)
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

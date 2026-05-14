"""Phase-timing profiler for the SGLang E2E benchmark.

Picks one scenario × set of methods, runs tree_batch_decode with
return_phase_timings=True, and prints per-phase mean/p50/p99 + total
contribution to decode time. Also dumps bs_kernel's sub-phase plan
trace if BS_KERNEL_TRACE_PLAN=1 is set (the bs_kernel driver appends
to a list on the backend).

Usage:
    BS_KERNEL_TRACE_PLAN=1 \\
    uv run python benchmarks/bs_kernel/profile_sglang_e2e.py \\
        --scenario multi_chain_reasoning \\
        --methods bs_kernel paged \\
        --max-new 256
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import torch
from transformers import AutoTokenizer

from beam_engine.baselines.fasttree import FastTreeBackend
from beam_engine.baselines.mlca import MlcaBackend
from beam_engine.baselines.paged import PagedBackend
from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
from beam_engine.methods.bs_kernel.driver import BsKernelBackend
from beam_engine.models.modeling_llama import LlamaForCausalLM
from beam_engine.tree_driver import tree_batch_decode

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from sglang_workloads import SCENARIO_BUILDERS


MODEL_NAME = os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")
DEVICE = "cuda"
DTYPE = torch.float16


def _summary(xs: list[float]) -> tuple[float, float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0, 0.0
    s = sorted(xs)
    return (
        sum(xs) / len(xs),
        statistics.median(xs),
        s[int(0.99 * (len(xs) - 1))],
        sum(xs),
    )


def _make_backend(name: str):
    if name == "bs_kernel":
        coeff = load_or_defaults(torch.device(DEVICE))
        return BsKernelBackend(coefficients=coeff)
    return {
        "fasttree": FastTreeBackend,
        "paged":    PagedBackend,
        "mlca":     MlcaBackend,
    }[name]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True,
                    choices=list(SCENARIO_BUILDERS.keys()))
    ap.add_argument("--methods", nargs="+", default=["bs_kernel", "paged"])
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    print(f"model: {MODEL_NAME}")
    print(f"scenario: {args.scenario}")
    print(f"methods:  {args.methods}")
    print(f"max_new:  {args.max_new}\n")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config

    spec = SCENARIO_BUILDERS[args.scenario](tok)
    spec.validate()
    print(f"B={spec.B} K={spec.K} n_leaves={spec.n_leaves}\n")

    for mname in args.methods:
        # 1 warmup, 1 measure.
        gc.collect(); torch.cuda.empty_cache()
        backend = _make_backend(mname)
        _ = tree_batch_decode(
            model, config, spec, args.max_new,
            backend=backend, return_timings=True,
        )
        del backend; gc.collect(); torch.cuda.empty_cache()

        backend = _make_backend(mname)
        t0 = time.perf_counter()
        result = tree_batch_decode(
            model, config, spec, args.max_new,
            backend=backend,
            return_timings=True,
            return_phase_timings=True,
        )
        wall = time.perf_counter() - t0
        t = result.timings
        decode_steps = t["decode_step_ms"]
        d_total = sum(decode_steps)

        print(f"=== {mname} ===")
        print(
            f"  wall {wall*1000:.1f}ms  "
            f"prefill_shared={t['prefill_shared_ms']:.1f}ms  "
            f"prefill_private={t['prefill_private_ms']:.1f}ms"
        )
        print(f"  decode_total = {d_total:.1f}ms  over {len(decode_steps)} steps")
        print(f"  {'phase':<14} {'mean':>8} {'p50':>8} {'p99':>8} {'total':>10}  {'% decode':>9}")
        for phase in ("alloc_ms", "plan_ms", "forward_ms", "topk_ms"):
            xs = t.get(phase, [])
            mean, p50, p99, total = _summary(xs)
            pct = (total / d_total * 100) if d_total > 0 else 0.0
            print(
                f"  {phase:<14} {mean:>8.3f} {p50:>8.3f} "
                f"{p99:>8.3f} {total:>10.2f}  {pct:>8.1f}%"
            )
        # Step-level p99/p50 view.
        sd = sorted(decode_steps)
        print(
            f"  step (overall) p50={statistics.median(decode_steps):.3f}ms  "
            f"p99={sd[int(0.99 * (len(sd) - 1))]:.3f}ms  "
            f"max={max(decode_steps):.3f}ms"
        )

        # Sub-phase plan trace (when {BS_KERNEL,FT,PAGED}_TRACE_PLAN=1).
        if hasattr(backend, "_plan_trace"):
            tr = backend._plan_trace
            if tr:
                print(f"\n  {mname} plan-trace: {len(tr)} steps")
                fields = sorted({
                    k for entry in tr
                    for k in entry.keys()
                    if isinstance(entry.get(k), (int, float))
                })
                for f in fields:
                    xs = [
                        float(e[f]) for e in tr
                        if isinstance(e.get(f), (int, float))
                    ]
                    if not xs:
                        continue
                    if f.endswith("_ms"):
                        mean, p50, p99, total = _summary(xs)
                        print(
                            f"    {f:<22} mean={mean:>7.3f}ms  "
                            f"p50={p50:>7.3f}  p99={p99:>7.3f}  "
                            f"total={total:>9.2f}"
                        )
                    else:
                        print(
                            f"    {f:<22} mean={sum(xs)/len(xs):>7.2f}  "
                            f"min={min(xs):>5}  max={max(xs):>5}"
                        )
        print()

        del backend, result; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

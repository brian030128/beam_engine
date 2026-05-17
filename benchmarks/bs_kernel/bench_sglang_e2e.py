"""End-to-end SGLang tree benchmark.

Mirrors the kernel-level test
(``benchmarks/bs_kernel/bench_sglang_tree_shapes.py``) but runs the
full inference loop on real text — prefill + max_new=256 greedy decode
steps for the four SGLang multi-* scenarios, compared across
bs_kernel, fasttree, paged, and mlca.

Usage:
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_level_system multi_few_shot \
                    multi_chain_reasoning multi_document \
        --methods bs_kernel fasttree paged mlca \
        --max-new 256 \
        --repeat 3 \
        --out logs/sglang_e2e.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from beam_engine.baselines.deft import DeftBackend
from beam_engine.baselines.fasttree import FastTreeBackend
from beam_engine.baselines.mlca import MlcaBackend
from beam_engine.baselines.paged import PagedBackend
from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
from beam_engine.methods.bs_kernel.cost_model import Strategy
from beam_engine.methods.bs_kernel.driver import BsKernelBackend
from beam_engine.models.modeling_llama import LlamaForCausalLM
from beam_engine.tree_driver import tree_batch_decode

from sglang_workloads import SCENARIO_BUILDERS, build_multi_chain_reasoning_stage2


MODEL_NAME = os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")
DEVICE = "cuda"
DTYPE = torch.float16


# bs_kernel forced-strategy aliases for picker-vs-kernel ablation.
# Each maps to a single Strategy that overrides the cost-model picker.
_BSK_FORCED: dict[str, Strategy] = {
    "bsk_per_beam":   Strategy.PER_BEAM,
    "bsk_2l_1p":      Strategy.SHARED_2L_1POOL,
    "bsk_3l_1p":      Strategy.SHARED_3L_1POOL,
    "bsk_2l_dt":      Strategy.SHARED_2L_DEC_TAIL,
    "bsk_3l_dt":      Strategy.SHARED_3L_DEC_TAIL,
    "bsk_2l_2p":      Strategy.SHARED_2L_2POOL,
}


def _make_backend(name: str):
    if name == "bs_kernel":
        coeff = load_or_defaults(torch.device(DEVICE))
        return BsKernelBackend(coefficients=coeff)
    if name in _BSK_FORCED:
        coeff = load_or_defaults(torch.device(DEVICE))
        return BsKernelBackend(
            coefficients=coeff,
            available_strategies={_BSK_FORCED[name]},
        )
    return {
        "fasttree": FastTreeBackend,
        "paged":    PagedBackend,
        "mlca":     MlcaBackend,
        "deft":     DeftBackend,
    }[name]()


BACKEND_FACTORIES: dict[str, object] = {
    "bs_kernel": None,  # handled by _make_backend
    "fasttree":  FastTreeBackend,
    "paged":     PagedBackend,
    "mlca":      MlcaBackend,
    "deft":      DeftBackend,
}
for _name in _BSK_FORCED:
    BACKEND_FACTORIES[_name] = None  # also routed through _make_backend


@dataclass
class Row:
    scenario: str
    method: str
    repeat_idx: int
    k_override: int   # 0 means "natural / unset"; otherwise the K used
    n_leaves: int
    prefill_shared_ms: float
    prefill_private_ms: float
    decode_total_ms: float
    decode_p50_ms: float
    decode_p99_ms: float
    n_decode_steps: int
    # Per-phase decode-step timings (return_phase_timings=True).
    plan_mean_ms: float
    plan_p50_ms: float
    plan_p99_ms: float
    plan_total_ms: float
    forward_mean_ms: float
    forward_p50_ms: float
    forward_p99_ms: float
    forward_total_ms: float
    alloc_mean_ms: float
    alloc_total_ms: float
    topk_mean_ms: float
    topk_total_ms: float
    # Total dispatch-decision time across all decode steps. 0 for
    # methods with no dispatch (paged, deft, mlca); for fasttree this
    # is the heuristic + parallelism convergence loop sum, for
    # bs_kernel it's the cost-model pick_strategy_batch sum. Requires
    # the relevant TRACE_PLAN env var to be on (otherwise stays 0).
    # build_indices_ms = plan_total_ms - dispatch_total_ms.
    dispatch_total_ms: float = 0.0


def _phase_stats(xs: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, p50, p99, total) for a per-step list of ms values."""
    if not xs:
        return 0.0, 0.0, 0.0, 0.0
    total = sum(xs)
    mean = total / len(xs)
    p50 = statistics.median(xs)
    sorted_xs = sorted(xs)
    p99 = sorted_xs[int(0.99 * (len(xs) - 1))]
    return mean, p50, p99, total


def _merge_timings(t1: dict, t2: dict) -> dict:
    """Combine the per-stage ``tree_batch_decode`` timing dicts of a
    two-stage workload (e.g. FastTree's multi_chain_reasoning fork+merge)
    into a single dict whose decode_step_ms / plan_ms / forward_ms /
    alloc_ms / topk_ms are concatenations and whose prefill_* are sums.
    """
    out: dict = {}
    for k in ("prefill_shared_ms", "prefill_private_ms"):
        out[k] = float(t1.get(k, 0.0)) + float(t2.get(k, 0.0))
    for k in ("decode_step_ms", "plan_ms", "forward_ms", "alloc_ms", "topk_ms"):
        out[k] = list(t1.get(k, [])) + list(t2.get(k, []))
    return out


def _run_one(
    model, config, spec, method_name: str, max_new: int,
    *, two_stage_spec=None, tokenizer=None,
) -> tuple[Row, list[float]]:
    backend = _make_backend(method_name)
    result = tree_batch_decode(
        model, config, spec, max_new,
        backend=backend,
        return_timings=True,
        return_phase_timings=True,
    )
    # Dump fasttree plan-trace if FT_TRACE_PLAN=1 (this path bypasses the
    # standalone ``fasttree.beam_search`` wrapper that normally dumps).
    if method_name == "fasttree" and os.environ.get("FT_TRACE_PLAN", "0") != "0":
        from beam_engine.baselines.fasttree import _dump_ft_plan_trace
        _dump_ft_plan_trace(backend._plan_trace)
    # Same for bs_kernel — BS_KERNEL_TRACE_PLAN=1.
    if method_name == "bs_kernel" and os.environ.get("BS_KERNEL_TRACE_PLAN", "0") != "0":
        from beam_engine.methods.bs_kernel.driver import _dump_plan_trace
        _dump_plan_trace(backend._plan_trace)
    # Same for mlca — MLCA_TRACE_PLAN=1.
    if method_name == "mlca" and os.environ.get("MLCA_TRACE_PLAN", "0") != "0":
        from beam_engine.baselines.mlca import _dump_mlca_plan_trace
        _dump_mlca_plan_trace(backend._plan_trace)
    # MLCA SM90 plan-cache hit/miss counter (whenever MLCA_PLAN_CACHE=1).
    if method_name == "mlca" and os.environ.get("MLCA_PLAN_CACHE", "0") != "0":
        h = getattr(backend, "_plan_hits", 0)
        m = getattr(backend, "_plan_misses", 0)
        tot = h + m
        rate = (h / tot * 100.0) if tot else 0.0
        print(f"[mlca plan-cache] hits={h} misses={m} hit-rate={rate:.1f}%")
    # Same for paged — PAGED_TRACE_PLAN=1.
    if method_name == "paged" and os.environ.get("PAGED_TRACE_PLAN", "0") != "0":
        from beam_engine.baselines.paged import _dump_paged_plan_trace
        _dump_paged_plan_trace(backend._plan_trace)
    t = result.timings
    # Optional stage-2 (FastTree multi_chain_reasoning fork-join majority
    # vote): re-run tree_batch_decode with a stage-2 TreeSpec on a fresh
    # backend, then concatenate the timing streams so the reported
    # decode_total / plan / forward sums cover BOTH stages.
    if two_stage_spec is not None:
        backend2 = _make_backend(method_name)
        result2 = tree_batch_decode(
            model, config, two_stage_spec, max_new,
            backend=backend2,
            return_timings=True,
            return_phase_timings=True,
        )
        t = _merge_timings(t, result2.timings)
    decode_steps = t["decode_step_ms"]
    decode_total = sum(decode_steps)
    sorted_d = sorted(decode_steps)
    p50 = statistics.median(decode_steps) if decode_steps else 0.0
    p99 = (
        sorted_d[int(0.99 * (len(decode_steps) - 1))]
        if decode_steps else 0.0
    )
    plan_mean, plan_p50, plan_p99, plan_total = _phase_stats(t.get("plan_ms", []))
    fwd_mean, fwd_p50, fwd_p99, fwd_total = _phase_stats(t.get("forward_ms", []))
    alloc_mean, _, _, alloc_total = _phase_stats(t.get("alloc_ms", []))
    topk_mean, _, _, topk_total = _phase_stats(t.get("topk_ms", []))
    # Dispatch-decision total across all decode steps. fasttree's
    # _build_metadata writes per-step 'dispatch_ms' into _plan_trace
    # (heuristic + parallelism convergence loop, mostly amortised by
    # the heuristic cache). bs_kernel's per-step cost-model pick is
    # logged as 'pick_ms' across all three dispatch sites (PER_BEAM,
    # prefix-prefill, shared). paged / mlca / deft have no dispatch
    # decision in the plan path → 0. Requires trace flag on for
    # fasttree (FT_TRACE_PLAN=1) / bs_kernel (BS_KERNEL_TRACE_PLAN=1);
    # trace adds torch.cuda.synchronize() calls so absolute plan
    # totals here are inflated vs the no-trace timing path. The
    # build_indices / dispatch ratio is the meaningful number.
    dispatch_total_ms = 0.0
    if method_name == "fasttree":
        for e in getattr(backend, "_plan_trace", []):
            dispatch_total_ms += float(e.get("dispatch_ms", 0.0))
    elif method_name == "bs_kernel":
        for e in getattr(backend, "_plan_trace", []):
            dispatch_total_ms += float(e.get("pick_ms", 0.0))
    row = Row(
        scenario="",  # filled in by caller
        method=method_name,
        repeat_idx=0,  # filled in by caller
        k_override=0,  # filled in by caller
        n_leaves=spec.n_leaves,
        prefill_shared_ms=float(t["prefill_shared_ms"]),
        prefill_private_ms=float(t["prefill_private_ms"]),
        decode_total_ms=decode_total,
        decode_p50_ms=p50,
        decode_p99_ms=p99,
        n_decode_steps=len(decode_steps),
        plan_mean_ms=plan_mean, plan_p50_ms=plan_p50, plan_p99_ms=plan_p99, plan_total_ms=plan_total,
        forward_mean_ms=fwd_mean, forward_p50_ms=fwd_p50, forward_p99_ms=fwd_p99, forward_total_ms=fwd_total,
        alloc_mean_ms=alloc_mean, alloc_total_ms=alloc_total,
        topk_mean_ms=topk_mean, topk_total_ms=topk_total,
        dispatch_total_ms=dispatch_total_ms,
    )
    return row, decode_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenarios", nargs="+",
        default=list(SCENARIO_BUILDERS.keys()),
        choices=list(SCENARIO_BUILDERS.keys()),
    )
    ap.add_argument(
        "--methods", nargs="+",
        default=list(BACKEND_FACTORIES.keys()),
        choices=list(BACKEND_FACTORIES.keys()),
    )
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--warmup", action="store_true",
                    help="discard the first iteration of each method")
    ap.add_argument(
        "--k-override", default="",
        help="comma-separated list of K values to override per scenario "
             "(e.g. '16,32'). Empty (default) uses each scenario's natural "
             "K. Each value spawns its own set of TreeSpecs; rows are tagged "
             "with the K used so phase-0 vs phase-1 cells can be joined.",
    )
    ap.add_argument(
        "--b-override", type=int, default=None,
        help="override B (number of prompt groups) for every scenario. "
             "Default = each scenario's natural B. Use to scale total "
             "leaves (e.g. --b-override 13 --k-override 16 → 208 leaves).",
    )
    ap.add_argument(
        "--paper-exact", action="store_true",
        help="Use FastTree §4.1 paper-exact workload construction: "
             "natural rendered sys-prompt length (no pad), 20-shot "
             "fewshot bundles with non-overlapping stride, CoT-prefixed "
             "private prefixes for chain-reasoning, and Llama-3 paper "
             "segments for multi_document. Each scenario keeps its "
             "natural (B, K) unless overridden.",
    )
    args = ap.parse_args()
    if args.k_override:
        k_overrides: list[int | None] = [int(x) for x in args.k_override.split(",")]
    else:
        k_overrides = [None]

    print(f"model: {MODEL_NAME}")
    print(f"scenarios: {args.scenarios}")
    print(f"methods:   {args.methods}")
    print(f"max_new:   {args.max_new}, repeat: {args.repeat}\n")

    print("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    print(f"k_overrides: {k_overrides}")
    print("Building TreeSpecs (this tokenizes the inputs)...")
    # specs keyed by (scenario_name, k_override)
    specs: dict[tuple[str, int | None], object] = {}
    stage2_specs: dict[tuple[str, int | None], object] = {}
    for k_override in k_overrides:
        for name in args.scenarios:
            spec = SCENARIO_BUILDERS[name](
                tok, k_override=k_override, b_override=args.b_override,
                paper_exact=args.paper_exact,
            )
            spec.validate()
            specs[(name, k_override)] = spec
            L_s_mean = sum(len(g.shared_prefix_ids) for g in spec.groups) / spec.B
            L_p_mean = sum(
                len(g.private_prefix_ids_per_leaf[0]) for g in spec.groups
            ) / spec.B
            ko_tag = "natural" if k_override is None else f"K={k_override}"
            print(
                f"  {name:<24} [{ko_tag:<8}] B={spec.B:<3d} K={spec.K:<3d} "
                f"L_shared~{int(L_s_mean):<5d} L_priv~{int(L_p_mean):<5d} "
                f"n_leaves={spec.n_leaves}"
            )
            # Optional stage-2 spec for paper-exact multi_chain_reasoning.
            if args.paper_exact and name == "multi_chain_reasoning":
                s2 = build_multi_chain_reasoning_stage2(
                    tok, stage1_chain_len=args.max_new,
                    b_override=args.b_override,
                    num_chains=spec.K,  # match stage 1's chain count
                )
                s2.validate()
                stage2_specs[(name, k_override)] = s2
                L_s2 = sum(len(g.shared_prefix_ids) for g in s2.groups) / s2.B
                print(
                    f"  {name + ' (stage2)':<24} [{ko_tag:<8}] B={s2.B:<3d} K={s2.K:<3d} "
                    f"L_shared~{int(L_s2):<5d} L_priv~    0 n_leaves={s2.n_leaves}"
                )
    print()

    rows: list[Row] = []
    per_scenario_summary: list[str] = []
    for k_override in k_overrides:
        for sname in args.scenarios:
            spec = specs[(sname, k_override)]
            ko_tag = "natural" if k_override is None else f"K={k_override}"
            print(f"--- {sname} [{ko_tag}] ---")
            per_method_best: dict[str, float] = {}
            for mname in args.methods:
                iters = args.repeat + (1 if args.warmup else 0)
                best_total = float("inf")
                for r in range(iters):
                    try:
                        gc.collect()
                        torch.cuda.empty_cache()
                        t0 = time.perf_counter()
                        row, _decode = _run_one(
                            model, config, spec, mname, args.max_new,
                            two_stage_spec=stage2_specs.get((sname, k_override)),
                            tokenizer=tok,
                        )
                        wall = time.perf_counter() - t0
                    except Exception as e:
                        print(f"  {mname:<12} r={r}  FAILED: {type(e).__name__}: {e}")
                        gc.collect()
                        torch.cuda.empty_cache()
                        break
                    gc.collect()
                    torch.cuda.empty_cache()
                    if args.warmup and r == 0:
                        print(f"  {mname:<12} warmup  wall={wall:5.1f}s  decode_total={row.decode_total_ms:7.1f}ms")
                        continue
                    effective_r = r - (1 if args.warmup else 0)
                    row.scenario = sname
                    row.repeat_idx = effective_r
                    row.k_override = 0 if k_override is None else int(k_override)
                    rows.append(row)
                    if row.decode_total_ms < best_total:
                        best_total = row.decode_total_ms
                    total_ms = (
                        row.prefill_shared_ms + row.prefill_private_ms + row.decode_total_ms
                    )
                    print(
                        f"  {mname:<12} r={effective_r}  "
                        f"prefill={row.prefill_shared_ms + row.prefill_private_ms:7.1f}ms  "
                        f"decode={row.decode_total_ms:7.1f}ms  "
                        f"plan/step={row.plan_mean_ms:5.2f}ms  "
                        f"fwd/step={row.forward_mean_ms:5.2f}ms  "
                        f"p50/step={row.decode_p50_ms:5.2f}ms  "
                        f"wall={wall:5.1f}s"
                    )
                per_method_best[mname] = best_total
            # Per-scenario ranking summary.
            if per_method_best:
                ranking = sorted(per_method_best.items(), key=lambda kv: kv[1])
                line = f"  rank (decode_total): " + ", ".join(
                    f"{m}={t:.1f}" for m, t in ranking
                )
                per_scenario_summary.append(f"{sname:<24} [{ko_tag:<8}] {line}")
                print(line)
            print()

    # CSV out.
    out = args.out
    if out is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        gpu = torch.cuda.get_device_name().replace(" ", "_")
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = results_dir / f"bench_sglang_e2e-{gpu}-{ts}.csv"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario", "k_override", "method", "repeat",
            "n_leaves", "prefill_shared_ms", "prefill_private_ms",
            "decode_total_ms", "decode_p50_ms", "decode_p99_ms",
            "n_decode_steps",
            "plan_mean_ms", "plan_p50_ms", "plan_p99_ms", "plan_total_ms",
            "forward_mean_ms", "forward_p50_ms", "forward_p99_ms", "forward_total_ms",
            "alloc_mean_ms", "alloc_total_ms",
            "topk_mean_ms", "topk_total_ms",
            "dispatch_total_ms",
        ])
        for r in rows:
            w.writerow([
                r.scenario, r.k_override, r.method, r.repeat_idx,
                r.n_leaves,
                f"{r.prefill_shared_ms:.4f}",
                f"{r.prefill_private_ms:.4f}",
                f"{r.decode_total_ms:.4f}",
                f"{r.decode_p50_ms:.4f}",
                f"{r.decode_p99_ms:.4f}",
                r.n_decode_steps,
                f"{r.plan_mean_ms:.4f}", f"{r.plan_p50_ms:.4f}",
                f"{r.plan_p99_ms:.4f}",  f"{r.plan_total_ms:.4f}",
                f"{r.forward_mean_ms:.4f}", f"{r.forward_p50_ms:.4f}",
                f"{r.forward_p99_ms:.4f}",  f"{r.forward_total_ms:.4f}",
                f"{r.alloc_mean_ms:.4f}",   f"{r.alloc_total_ms:.4f}",
                f"{r.topk_mean_ms:.4f}",    f"{r.topk_total_ms:.4f}",
                f"{r.dispatch_total_ms:.4f}",
            ])
    print(f"\nwrote {len(rows)} rows to {out}\n")

    print("=== summary (best decode_total per method, ms) ===")
    for line in per_scenario_summary:
        print(line)


if __name__ == "__main__":
    main()

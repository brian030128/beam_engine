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
from beam_engine.distributed import (
    destroy_tp,
    get_tp_rank,
    get_tp_world_size,
    init_tp,
)
from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
from beam_engine.methods.bs_kernel.cost_model import Strategy
from beam_engine.methods.bs_kernel.driver import BsKernelBackend
from beam_engine.models import load_model_for_causal_lm
from beam_engine.tree_driver import tree_batch_decode

from sglang_workloads import (
    SCENARIO_BUILDERS, build_multi_chain_reasoning_stage2, _pad_or_truncate,
)


MODEL_NAME = os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")
# Bound to ``cuda:LOCAL_RANK`` inside ``main`` once init_tp() runs. Plain
# ``cuda`` is fine for single-GPU runs (no torchrun → LOCAL_RANK unset).
DEVICE = "cuda"
# Compute dtype env-driven so 70B-fp8 cells (which need bf16 activations
# to avoid fp16 attention-score overflow at long L_p) don't have to
# fork the bench. Default fp16 matches the historic 1B / 8B paper runs.
_BE_DTYPE = os.environ.get("BE_DTYPE", "fp16").lower()
DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}[_BE_DTYPE]


# bs_kernel forced-strategy aliases for picker-vs-kernel ablation.
# Each maps to a single Strategy that overrides the cost-model picker.
_BSK_FORCED: dict[str, Strategy] = {
    "bsk_per_beam":     Strategy.PER_BEAM,
    "bsk_2l_1p":        Strategy.SHARED_2L_1POOL,
    "bsk_2l_1p_single": Strategy.SHARED_2L_1POOL,
    "bsk_3l_1p":        Strategy.SHARED_3L_1POOL,
    "bsk_2l_dt":        Strategy.SHARED_2L_DEC_TAIL,
    "bsk_3l_dt":        Strategy.SHARED_3L_DEC_TAIL,
    "bsk_2l_2p":        Strategy.SHARED_2L_2POOL,
    # Force the cross-prompt 3L dec_tail (only realizes when the batch shares a
    # cross-prompt sys prefix, i.e. multi_few_shot --tree-root with B>=2);
    # otherwise the picker falls back to per_beam. For A/B vs the picker.
    "bsk_xp_dt":        Strategy.XPROMPT_DEC_TAIL,
}

# Aliases that need ``single_launch=True`` patched onto every
# FusedMultiLevelCascadeAttentionWrapper.__init__ for the duration of the
# run. Patch is process-global; only enable in a process that runs ONLY
# the patched alias to avoid contaminating other backends.
_BSK_SINGLE_LAUNCH: set[str] = {"bsk_2l_1p_single"}

# bs_kernel step-scheduled aliases (ablation): instead of letting the
# cost-model picker choose per step, force a hand-picked strategy crossover
# keyed on the decode-step index. Each maps to a list of
# ``(start_step, Strategy)`` thresholds; the strategy with the greatest
# ``start_step <= step`` is forced (last applicable persists to end-of-run).
# The switch step is env-overridable via ``BE_BSK_SCHED_SWITCH`` (default
# 2000) so the crossover can be probed without a code edit. ``bsk_sched_1p_dt``
# = SHARED_2L_1POOL for the first ``BE_BSK_SCHED_SWITCH`` decode steps, then
# SHARED_2L_DEC_TAIL for the rest (1POOL is single-launch when
# BE_PICKER_SINGLE_LAUNCH=1, matching the bsk_2l_1p_single baseline).
_SCHED_SWITCH = int(os.environ.get("BE_BSK_SCHED_SWITCH", "2000"))
_BSK_SCHEDULED: dict[str, list[tuple[int, Strategy]]] = {
    "bsk_sched_1p_dt": [
        (0, Strategy.SHARED_2L_1POOL),
        (_SCHED_SWITCH, Strategy.SHARED_2L_DEC_TAIL),
    ],
}


def _maybe_patch_picker_single_launch() -> None:
    """If ``BE_PICKER_SINGLE_LAUNCH=1``, make every
    FusedMultiLevelCascadeAttentionWrapper default to ``single_launch=True``
    process-wide. This lets the cost-model picker (``bs_kernel``) realize
    the single-launch SHARED_2L_1POOL win when it selects that strategy.
    Safe to run alongside paged/mlca/fasttree/deft — none of them build
    that wrapper, so the patch never fires for them. The forced 1POOL
    aliases simply become single-launch too (idempotent)."""
    if os.environ.get("BE_PICKER_SINGLE_LAUNCH", "0") in ("0", "", "false"):
        return
    import flashinfer
    _orig_init = flashinfer.FusedMultiLevelCascadeAttentionWrapper.__init__
    if getattr(_orig_init, "_be_single_launch_patched", False):
        return

    def _patched_init(self, *a, **kw):
        kw.setdefault("single_launch", True)
        return _orig_init(self, *a, **kw)

    _patched_init._be_single_launch_patched = True
    flashinfer.FusedMultiLevelCascadeAttentionWrapper.__init__ = _patched_init


def _maybe_force_fa2() -> None:
    """If ``BE_FORCE_FA2=1``, force flashinfer's prefill backend auto-selection
    to FA2 process-wide. Needed on head_dim=64 (1B): the current flashinfer
    fork's FA3/SM90 ``batch_prefill`` kernel fails to COMPILE for hd=64
    (cutlass "No eligible GMMA operator" for the 192x96 tile), which takes down
    paged / fasttree / tree / mlca (anything that builds that op) the moment a
    cold rebuild is triggered. FA2 (SM80) compiles and runs for all head dims
    on Hopper. ``determine_attention_backend`` (called by the prefill wrappers
    and by mlca's per-level wrappers when backend='auto') has no env hook, so
    patch it. Scope to 1B via the env so 8B/70B (hd=128, FA3 compiles) keep FA3.
    Idempotent; orthogonal to the single_launch patch."""
    if os.environ.get("BE_FORCE_FA2", "0") in ("0", "", "false"):
        return
    import flashinfer.prefill as _fip
    import flashinfer.utils as _fiu
    if getattr(_fiu.determine_attention_backend, "_be_fa2_forced", False):
        return

    def _force_fa2(*a, **kw):
        return "fa2"

    _force_fa2._be_fa2_forced = True
    # Patch both the source (utils) and the name already imported into prefill.
    _fiu.determine_attention_backend = _force_fa2
    _fip.determine_attention_backend = _force_fa2


def _make_backend(name: str):
    if name == "bs_kernel":
        coeff = load_or_defaults(torch.device(DEVICE), MODEL_NAME, tp_size=get_tp_world_size())
        return BsKernelBackend(coefficients=coeff)
    if name in _BSK_FORCED:
        if name in _BSK_SINGLE_LAUNCH:
            import flashinfer
            _orig_init = flashinfer.FusedMultiLevelCascadeAttentionWrapper.__init__
            if not getattr(_orig_init, "_be_single_launch_patched", False):
                def _patched_init(self, *a, **kw):
                    kw.setdefault("single_launch", True)
                    return _orig_init(self, *a, **kw)
                _patched_init._be_single_launch_patched = True
                flashinfer.FusedMultiLevelCascadeAttentionWrapper.__init__ = _patched_init
        coeff = load_or_defaults(torch.device(DEVICE), MODEL_NAME, tp_size=get_tp_world_size())
        return BsKernelBackend(
            coefficients=coeff,
            available_strategies={_BSK_FORCED[name]},
        )
    if name in _BSK_SCHEDULED:
        coeff = load_or_defaults(torch.device(DEVICE), MODEL_NAME, tp_size=get_tp_world_size())
        return BsKernelBackend(
            coefficients=coeff,
            strategy_schedule=_BSK_SCHEDULED[name],
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
for _name in _BSK_SCHEDULED:
    BACKEND_FACTORIES[_name] = None  # step-scheduled, routed through _make_backend


@dataclass
class Row:
    scenario: str
    method: str
    repeat_idx: int
    k_override: int   # 0 means "natural / unset"; otherwise the K used
    data_seed: int    # which data draw (0 = the reference / original draw)
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


def _spec_ref_shape(spec):
    """Reference shape of a 2-level (PromptGroup) spec: per group, the
    (shared_len, private_len) token counts. Private prefixes are uniform
    within a group (TreeSpec.validate), so one private_len per group
    suffices. Returns None for multilevel specs (those are already padded
    to a uniform leaf depth by the builder)."""
    if spec.is_multilevel:
        return None
    return [
        (len(g.shared_prefix_ids), len(g.private_prefix_ids_per_leaf[0]))
        for g in spec.groups
    ]


def _conform_spec_to_ref(spec, ref) -> None:
    """Pad/truncate every group's shared + private token lists to the
    reference shape (in place). This makes a resampled data draw feed the
    model the *exact same token-length shape* as the seed-0 draw, so only
    the text content differs across draws — attention timing (which depends
    on token counts, not content) is held fixed. No-op when ``ref`` is None
    (multilevel) or already matches."""
    if ref is None:
        return
    if len(spec.groups) != len(ref):
        raise ValueError(
            f"conform: group count {len(spec.groups)} != reference {len(ref)}"
        )
    for g, (shared_len, priv_len) in zip(spec.groups, ref):
        g.shared_prefix_ids = _pad_or_truncate(g.shared_prefix_ids, shared_len)
        g.private_prefix_ids_per_leaf = [
            _pad_or_truncate(p, priv_len) for p in g.private_prefix_ids_per_leaf
        ]


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
    *, two_stage_spec=None, tokenizer=None, scenario_name: str = "",
) -> tuple[Row, list[float]]:
    backend = _make_backend(method_name)
    _kv_dtype_env = os.environ.get("BE_KV_DTYPE", "").lower()
    _kv_dtype = {
        "": None, "fp16": torch.float16, "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn, "fp8_e4m3": torch.float8_e4m3fn,
    }.get(_kv_dtype_env)
    result = tree_batch_decode(
        model, config, spec, max_new,
        backend=backend,
        return_timings=True,
        return_phase_timings=True,
        dtype=DTYPE,
        kv_dtype=_kv_dtype,
    )
    # Optional beam dump for cross-method equality checks
    # (BE_DUMP_BEAMS_DIR=<dir> → writes <dir>/<scenario>_<method>.json with
    # leaf_token_ids). Only rank-0 writes under torchrun.
    _dump_dir = os.environ.get("BE_DUMP_BEAMS_DIR", "")
    if _dump_dir and (not torch.distributed.is_initialized()
                      or torch.distributed.get_rank() == 0):
        import json
        os.makedirs(_dump_dir, exist_ok=True)
        tag = scenario_name or f"B{spec.B}K{spec.K}"
        path = os.path.join(_dump_dir, f"{tag}__{method_name}.json")
        with open(path, "w") as f:
            json.dump({
                "scenario": scenario_name,
                "method": method_name,
                "B": spec.B, "K": spec.K,
                "leaf_token_ids": result.leaf_token_ids,
            }, f)
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
            dtype=DTYPE,
            kv_dtype=_kv_dtype,
        )
        t = _merge_timings(t, result2.timings)
    # Optional per-step timing dump (BE_DUMP_STEP_TIMINGS=<dir>): write the
    # full per-decode-step plan/forward/total streams so the 1POOL-vs-DEC_TAIL
    # forward-time-vs-tail-length crossover can be analysed offline. One CSV
    # per (scenario, method); last (measured) repeat wins. Rank-0 only.
    _step_dir = os.environ.get("BE_DUMP_STEP_TIMINGS", "")
    if _step_dir and (not torch.distributed.is_initialized()
                      or torch.distributed.get_rank() == 0):
        os.makedirs(_step_dir, exist_ok=True)
        tag = scenario_name or f"B{spec.B}K{spec.K}"
        spath = os.path.join(_step_dir, f"{tag}__{method_name}.csv")
        _ds = t.get("decode_step_ms", [])
        _pl = t.get("plan_ms", [])
        _fw = t.get("forward_ms", [])
        # Per-step picked strategy (bs_kernel / forced / scheduled backends).
        _st = getattr(backend, "_picked_strategy_log", [])
        with open(spath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "decode_step_ms", "plan_ms", "forward_ms", "strategy"])
            for i in range(len(_ds)):
                w.writerow([
                    i,
                    _ds[i],
                    _pl[i] if i < len(_pl) else "",
                    _fw[i] if i < len(_fw) else "",
                    _st[i] if i < len(_st) else "",
                ])
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
        data_seed=0,   # filled in by caller
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


def _write_results_csv(out_path, rows) -> None:
    """Write all accumulated rows to ``out_path`` (full rewrite). Called
    incrementally after each method so partial results survive a timeout
    (e.g. a slow deft method on a long-decode 70B cell) — the bench used to
    write only once at the very end, losing everything on a SLURM time-out."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario", "k_override", "method", "repeat", "data_seed",
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
                r.scenario, r.k_override, r.method, r.repeat_idx, r.data_seed,
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


def main():
    _maybe_patch_picker_single_launch()
    _maybe_force_fa2()
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
    ap.add_argument(
        "--data-seeds", default="0",
        help="comma-separated list of data-draw seeds (e.g. '0,1,2'). Each "
             "seed resamples the underlying GSM8K/HotpotQA text for the SAME "
             "workload shape: seed 0 is the reference draw (reproduces the "
             "single-draw numbers), and every other seed's spec is conformed "
             "to seed 0's per-group token lengths so the model input shape is "
             "identical across draws. Rows are tagged with data_seed.",
    )
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
    ap.add_argument(
        "--no-stage2", action="store_true",
        help="Disable the multi_chain_reasoning stage-2 majority-vote "
             "decode. Useful for picker-vs-forced ablations where the "
             "K=1 stage-2 muddies the per-strategy comparison.",
    )
    ap.add_argument(
        "--tree-root", action="store_true",
        help="Build workloads as multi-level TreeSpec.root (sys → fewshot "
             "→ QA leaves) instead of the legacy 2-level PromptGroup form. "
             "Cross-group sys-sharing becomes visible at the page-ID level "
             "so fasttree/deft's combined-radix builders discover the "
             "depth-3 sharing; bs_kernel still picks (sys+fewshot, output) "
             "per group at depth=2 (cross-prompt picker is a follow-up). "
             "Only the builders that exposes 3-level structure honor it "
             "today (multi_few_shot; others passthrough).",
    )
    args = ap.parse_args()
    if args.k_override:
        k_overrides: list[int | None] = [int(x) for x in args.k_override.split(",")]
    else:
        k_overrides = [None]
    data_seeds: list[int] = [int(x) for x in args.data_seeds.split(",") if x.strip() != ""]
    if not data_seeds:
        data_seeds = [0]
    # The first seed is the reference draw whose token-length shape every
    # other draw is conformed to. Keep it first (seed 0 by convention).
    ref_seed = data_seeds[0]

    # Bind to the per-rank device under torchrun; a no-op for single-GPU
    # invocation (init_tp short-circuits when WORLD_SIZE=1).
    init_tp()
    tp_rank = get_tp_rank()
    tp_size = get_tp_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global DEVICE
    DEVICE = f"cuda:{local_rank}" if tp_size > 1 else "cuda"
    is_rank0 = tp_rank == 0

    def _say(*a, **kw):
        if is_rank0:
            print(*a, **kw)

    _say(f"model: {MODEL_NAME}  (tp_size={tp_size})")
    _say(f"scenarios: {args.scenarios}")
    _say(f"methods:   {args.methods}")
    _say(f"max_new:   {args.max_new}, repeat: {args.repeat}\n")

    _say("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model_for_causal_lm(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    _say("Model loaded.\n")

    _say(f"k_overrides: {k_overrides}")
    _say(f"data_seeds: {data_seeds}  (seed {ref_seed} = reference shape)")
    _say("Building TreeSpecs (this tokenizes the inputs)...")
    # specs keyed by (scenario_name, k_override, data_seed). Seed ref_seed is
    # built first and its per-group token-length shape becomes the reference
    # that every other seed is conformed to (so all draws feed the model the
    # same shape; only the text content differs).
    import inspect
    specs: dict[tuple[str, int | None, int], object] = {}
    stage2_specs: dict[tuple[str, int | None, int], object] = {}
    for k_override in k_overrides:
        for name in args.scenarios:
            builder = SCENARIO_BUILDERS[name]
            accepts = inspect.signature(builder).parameters
            ko_tag = "natural" if k_override is None else f"K={k_override}"
            ref_shape = None
            ref_shape_s2 = None
            for si, seed in enumerate(data_seeds):
                builder_kwargs: dict = dict(
                    k_override=k_override, b_override=args.b_override,
                    paper_exact=args.paper_exact,
                )
                # Only builders that grew these kwargs accept them; the
                # others fall back to the legacy form.
                if "tree_root" in accepts:
                    builder_kwargs["tree_root"] = args.tree_root
                if "data_seed" in accepts:
                    builder_kwargs["data_seed"] = seed
                spec = builder(tok, **builder_kwargs)
                if si == 0:
                    ref_shape = _spec_ref_shape(spec)
                else:
                    _conform_spec_to_ref(spec, ref_shape)
                spec.validate()
                specs[(name, k_override, seed)] = spec

                # Optional stage-2 spec for paper-exact multi_chain_reasoning.
                if args.paper_exact and name == "multi_chain_reasoning" and not args.no_stage2:
                    s2 = build_multi_chain_reasoning_stage2(
                        tok, stage1_chain_len=args.max_new,
                        b_override=args.b_override,
                        num_chains=spec.K,  # match stage 1's chain count
                    )
                    if si == 0:
                        ref_shape_s2 = _spec_ref_shape(s2)
                    else:
                        _conform_spec_to_ref(s2, ref_shape_s2)
                    s2.validate()
                    stage2_specs[(name, k_override, seed)] = s2

            # Log the reference draw's shape (identical for every seed after
            # conforming) plus the draw count.
            spec0 = specs[(name, k_override, ref_seed)]
            if spec0.is_multilevel:
                from beam_engine.tree_driver import _enumerate_leaf_parents
                lps, anc_paths = _enumerate_leaf_parents(spec0.root)
                L_s_mean = sum(
                    sum(len(a.token_ids) for a in path) + len(lp.token_ids)
                    for lp, path in zip(lps, anc_paths)
                ) / max(1, len(lps))
                L_p_mean = sum(
                    len(lp.children[0].token_ids) for lp in lps
                ) / max(1, len(lps))
            else:
                L_s_mean = sum(
                    len(g.shared_prefix_ids) for g in spec0.groups
                ) / spec0.B
                L_p_mean = sum(
                    len(g.private_prefix_ids_per_leaf[0]) for g in spec0.groups
                ) / spec0.B
            _say(
                f"  {name:<24} [{ko_tag:<8}] B={spec0.B:<3d} K={spec0.K:<3d} "
                f"L_shared~{int(L_s_mean):<5d} L_priv~{int(L_p_mean):<5d} "
                f"n_leaves={spec0.n_leaves}  draws={len(data_seeds)}"
            )
            if (name, k_override, ref_seed) in stage2_specs:
                s2 = stage2_specs[(name, k_override, ref_seed)]
                L_s2 = sum(len(g.shared_prefix_ids) for g in s2.groups) / s2.B
                _say(
                    f"  {name + ' (stage2)':<24} [{ko_tag:<8}] B={s2.B:<3d} K={s2.K:<3d} "
                    f"L_shared~{int(L_s2):<5d} L_priv~    0 n_leaves={s2.n_leaves}"
                )
    _say("")

    rows: list[Row] = []
    per_scenario_summary: list[str] = []
    for k_override in k_overrides:
        for sname in args.scenarios:
            ko_tag = "natural" if k_override is None else f"K={k_override}"
            _say(f"--- {sname} [{ko_tag}] ---")
            # best decode_total per method, taken across all data draws.
            per_method_best: dict[str, float] = {}
            for seed in data_seeds:
                spec = specs[(sname, k_override, seed)]
                if len(data_seeds) > 1:
                    _say(f"  [draw seed={seed}]")
                for mname in args.methods:
                    # Warm up only on the reference draw: every draw shares the
                    # same token-length shape (conformed), so the Triton-JIT /
                    # CUDA-graph capture absorbed by the warmup is already done
                    # by the time later seeds run. This keeps extra draws cheap.
                    do_warmup = args.warmup and seed == data_seeds[0]
                    iters = args.repeat + (1 if do_warmup else 0)
                    best_total = per_method_best.get(mname, float("inf"))
                    for r in range(iters):
                        try:
                            gc.collect()
                            torch.cuda.empty_cache()
                            t0 = time.perf_counter()
                            row, _decode = _run_one(
                                model, config, spec, mname, args.max_new,
                                scenario_name=sname,
                                two_stage_spec=stage2_specs.get((sname, k_override, seed)),
                                tokenizer=tok,
                            )
                            wall = time.perf_counter() - t0
                        except Exception as e:
                            import traceback as _tb
                            _say(f"  {mname:<12} seed={seed} r={r}  FAILED: {type(e).__name__}: {e}")
                            if os.environ.get("BE_TRACEBACK_ON_FAIL", "0") != "0":
                                _say(_tb.format_exc())
                            gc.collect()
                            torch.cuda.empty_cache()
                            break
                        gc.collect()
                        torch.cuda.empty_cache()
                        if do_warmup and r == 0:
                            _say(f"  {mname:<12} seed={seed} warmup  wall={wall:5.1f}s  decode_total={row.decode_total_ms:7.1f}ms")
                            continue
                        effective_r = r - (1 if do_warmup else 0)
                        row.scenario = sname
                        row.repeat_idx = effective_r
                        row.k_override = 0 if k_override is None else int(k_override)
                        row.data_seed = seed
                        rows.append(row)
                        if row.decode_total_ms < best_total:
                            best_total = row.decode_total_ms
                        total_ms = (
                            row.prefill_shared_ms + row.prefill_private_ms + row.decode_total_ms
                        )
                        _say(
                            f"  {mname:<12} seed={seed} r={effective_r}  "
                            f"prefill={row.prefill_shared_ms + row.prefill_private_ms:7.1f}ms  "
                            f"decode={row.decode_total_ms:7.1f}ms  "
                            f"plan/step={row.plan_mean_ms:5.2f}ms  "
                            f"fwd/step={row.forward_mean_ms:5.2f}ms  "
                            f"p50/step={row.decode_p50_ms:5.2f}ms  "
                            f"wall={wall:5.1f}s"
                        )
                    per_method_best[mname] = best_total
                    # Flush results so far so a later slow/hung method (deft on a
                    # long-decode 70B cell) timing out doesn't discard the methods
                    # that already completed.
                    if is_rank0 and args.out is not None:
                        _write_results_csv(args.out, rows)
            # Per-scenario ranking summary (best across all draws).
            if per_method_best:
                ranking = sorted(per_method_best.items(), key=lambda kv: kv[1])
                line = f"  rank (decode_total): " + ", ".join(
                    f"{m}={t:.1f}" for m, t in ranking
                )
                per_scenario_summary.append(f"{sname:<24} [{ko_tag:<8}] {line}")
                _say(line)
            _say("")

    # CSV out — only rank 0 writes; all ranks computed identical timings up
    # to TP all-reduce noise, and the model state is rank-symmetric.
    if is_rank0:
        out = args.out
        if out is None:
            results_dir = Path(__file__).parent / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            gpu = torch.cuda.get_device_name().replace(" ", "_")
            ts = time.strftime("%Y%m%d-%H%M%S")
            out = results_dir / f"bench_sglang_e2e-{gpu}-{ts}.csv"
        _write_results_csv(out, rows)
        print(f"\nwrote {len(rows)} rows to {out}\n")

        print("=== summary (best decode_total per method, ms) ===")
        for line in per_scenario_summary:
            print(line)

    destroy_tp()


if __name__ == "__main__":
    main()

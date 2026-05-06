"""End-to-end decode-latency sweep across all baselines + bs_kernel.

Drives the D1 ablation grid from the design plan (`docs/bs_kernel_design.md`):
end-to-end Llama beam-search decode latency vs (paged, tree, fasttree,
adaptive_pool, bs_kernel) across the (K, L_p, max_new) grid.

Output: CSV at ``benchmarks/bs_kernel/results/sweep-<gpu>-<timestamp>.csv``
with one row per (method, K, L_p, max_new) tuple, columns:

    method, K, L_p, max_new, prefill_ms,
    decode_step_ms_p50, decode_step_ms_p90, decode_step_ms_p99,
    decode_step_ms_mean, decode_step_count

Per CLAUDE.md: pick a fully-idle GPU; run via
``CUDA_VISIBLE_DEVICES=<id> uv run python benchmarks/bs_kernel/sweep.py``.

Use ``--quick`` for a small grid suitable for fast iteration; ``--full``
runs the design-doc grid (K ∈ {1,4,16,64} × L_p ∈ {128,4096} ×
max_new ∈ {16,128}). The full grid takes 30+ min on RTX-class GPUs.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import dbs, fasttree, mlca, paged, tree
from beam_engine.methods import adaptive_pool, bs_kernel
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16


# Method registry — name → callable matching the
# `beam_search(model, config, prompts, max_new, K, *, return_timings=True)`
# contract.
METHODS: dict[str, Callable] = {
    "paged":         paged.beam_search,
    "tree":          tree.beam_search,
    "fasttree":      fasttree.beam_search,
    "mlca":          mlca.beam_search,
    "adaptive_pool": adaptive_pool.beam_search,
    "bs_kernel":     bs_kernel.beam_search,
    "dbs":           dbs.beam_search,
}


# Grids — long-prefix regime (L_p ≥ 1024). Below this, sharing the
# prefix doesn't pay back the merge launch (see docs/bs_kernel_design.md
# § Decision 1) so the SHARED strategies the cost model picks are
# uninteresting; we only benchmark the regime where the design matters.
QUICK_GRID = [
    (16, 8192, 16),
    (32, 8192, 16),
    (64, 8192, 16),
]

FULL_GRID = [
    (K, L_p, 16)
    for K in (16, 32, 64)
    for L_p in (2048, 8192, 32768, 65536)
]


@dataclass
class Row:
    method: str
    K: int
    L_p: int
    max_new: int
    prefill_ms: float
    decode_step_ms_p50: float
    decode_step_ms_p90: float
    decode_step_ms_p99: float
    decode_step_ms_mean: float
    decode_step_count: int


def _make_prompt(tokenizer, target_len: int) -> list[int]:
    snippet = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
        "She traveled across mountains and rivers, recording her findings "
        "in a leather-bound journal that grew thicker with every passing "
        "season. The townsfolk welcomed her with warm hearts and cool drinks. "
    )
    # Iteratively grow until we have enough tokens — tokenization can be
    # non-monotonic across joins so we don't trust the per-snippet count.
    repeats = max(8, target_len // 50)
    while True:
        ids = tokenizer.encode(snippet * repeats, add_special_tokens=False)
        if len(ids) >= target_len:
            break
        repeats *= 2
    return ids[:target_len]


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def run_one(
    method_name: str,
    method_fn: Callable,
    model,
    config,
    prompt_ids: list[int],
    K: int,
    max_new: int,
) -> Row | None:
    import inspect

    # Sized for: prompt pages + per-beam suffix pages + safety headroom.
    # Page-based methods (paged, adaptive_pool, bs_kernel) need this
    # plumbed through; tree / fasttree allocate fused buffers internally.
    page_size = 16
    needed_pages = (
        (len(prompt_ids) + page_size - 1) // page_size
        + K * ((max_new + page_size) // page_size)
        + 256
    )
    extra_kwargs: dict = {}
    sig = inspect.signature(method_fn)
    if "max_num_pages" in sig.parameters:
        extra_kwargs["max_num_pages"] = needed_pages

    try:
        beams, timings = method_fn(
            model, config, [prompt_ids], max_new, K,
            return_timings=True, **extra_kwargs,
        )
    except Exception as e:
        print(f"  [{method_name}] FAILED: {type(e).__name__}: {e}")
        return None

    decode_steps = timings["decode_step_ms"]
    return Row(
        method=method_name,
        K=K, L_p=len(prompt_ids), max_new=max_new,
        prefill_ms=timings["prefill_ms"],
        decode_step_ms_p50=_percentile(decode_steps, 0.50),
        decode_step_ms_p90=_percentile(decode_steps, 0.90),
        decode_step_ms_p99=_percentile(decode_steps, 0.99),
        decode_step_ms_mean=(statistics.mean(decode_steps) if decode_steps else float("nan")),
        decode_step_count=len(decode_steps),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run the full design-doc grid")
    ap.add_argument("--quick", action="store_true", help="run a small grid (default)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                    help=f"subset of methods to run; default: all ({list(METHODS.keys())})")
    ap.add_argument("--out", default=None,
                    help="output CSV path; default auto-named under benchmarks/bs_kernel/results/")
    args = ap.parse_args()

    if args.full:
        grid = FULL_GRID
    else:
        grid = QUICK_GRID
    methods = {k: METHODS[k] for k in args.methods if k in METHODS}
    if not methods:
        print(f"no valid methods in {args.methods}; valid: {list(METHODS.keys())}", file=sys.stderr)
        sys.exit(2)

    print(f"grid: {len(grid)} configs × {len(methods)} methods = {len(grid)*len(methods)} runs")
    print(f"methods: {list(methods.keys())}")

    print("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    rows: list[Row] = []
    for (K, L_p, max_new) in grid:
        prompt_ids = _make_prompt(tok, L_p)
        for name, fn in methods.items():
            print(f"  {name:<14} K={K:<3d} L_p={L_p:<5d} max_new={max_new:<4d}", end=" ", flush=True)
            t0 = time.perf_counter()
            row = run_one(name, fn, model, config, prompt_ids, K, max_new)
            t1 = time.perf_counter()
            if row is not None:
                rows.append(row)
                print(f"prefill={row.prefill_ms:6.1f}ms  decode={row.decode_step_ms_p50:6.2f}ms p50  ({t1-t0:5.1f}s wall)")

    # Write CSV.
    out = args.out
    if out is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        gpu = torch.cuda.get_device_name().replace(" ", "_")
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = results_dir / f"sweep-{gpu}-{ts}.csv"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "K", "L_p", "max_new", "prefill_ms",
            "decode_step_ms_p50", "decode_step_ms_p90", "decode_step_ms_p99",
            "decode_step_ms_mean", "decode_step_count",
        ])
        for r in rows:
            w.writerow([
                r.method, r.K, r.L_p, r.max_new, f"{r.prefill_ms:.4f}",
                f"{r.decode_step_ms_p50:.4f}", f"{r.decode_step_ms_p90:.4f}",
                f"{r.decode_step_ms_p99:.4f}", f"{r.decode_step_ms_mean:.4f}",
                r.decode_step_count,
            ])
    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

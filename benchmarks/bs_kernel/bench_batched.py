"""Cross-kernel batched-decode benchmark.

For each method × (K, L_p, B), run beam search on B identical prompts and
report total decode wall time per (prompt, token). bs_kernel and paged are
natively batched (one launch per step covers all B prompts); tree, fasttree,
mlca, adaptive_pool sequentialize per prompt internally.

Output CSV columns:
    method, K, L_p, B, max_new, prefill_ms, decode_total_ms,
    decode_per_token_ms, decode_per_prompt_per_token_ms

Usage:
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 16 --L_p 8192 --B 1 2 4 8 --max_new 16 \
        --methods paged tree fasttree mlca adaptive_pool bs_kernel \
        --out benchmarks/bs_kernel/results/bench_batched.csv
"""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import fasttree, mlca, paged, tree
from beam_engine.methods import adaptive_pool, bs_kernel
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16


METHODS: dict[str, Callable] = {
    "paged":         paged.beam_search,
    "tree":          tree.beam_search,
    "fasttree":      fasttree.beam_search,
    "mlca":          mlca.beam_search,
    "adaptive_pool": adaptive_pool.beam_search,
    "bs_kernel":     bs_kernel.beam_search,
}


@dataclass
class Row:
    method: str
    K: int
    L_p: int
    B: int
    max_new: int
    prefill_ms: float
    decode_total_ms: float
    decode_per_token_ms: float
    decode_per_prompt_per_token_ms: float


def _make_prompt(tokenizer, target_len: int) -> list[int]:
    snippet = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
        "She traveled across mountains and rivers, recording her findings "
        "in a leather-bound journal that grew thicker with every passing "
        "season. "
    )
    repeats = max(8, target_len // 50)
    while True:
        ids = tokenizer.encode(snippet * repeats, add_special_tokens=False)
        if len(ids) >= target_len:
            break
        repeats *= 2
    return ids[:target_len]


def run_one(
    method_name: str,
    method_fn: Callable,
    model,
    config,
    prompts: list[list[int]],
    K: int,
    max_new: int,
) -> Row | None:
    B = len(prompts)
    L_p = len(prompts[0])
    page_size = 16
    needed_pages = (
        sum((len(p) + page_size - 1) // page_size for p in prompts)
        + B * K * ((max_new + page_size) // page_size + 2)
        + 256
    )
    extra_kwargs: dict = {}
    sig = inspect.signature(method_fn)
    if "max_num_pages" in sig.parameters:
        extra_kwargs["max_num_pages"] = needed_pages

    try:
        beams, timings = method_fn(
            model, config, prompts, max_new, K,
            return_timings=True, **extra_kwargs,
        )
    except Exception as e:
        print(f"  [{method_name}] FAILED: {type(e).__name__}: {e}")
        return None

    decode_steps = timings["decode_step_ms"]
    decode_total = sum(decode_steps)
    n_steps = len(decode_steps)
    # Tokens decoded across the whole call: each method runs (max_new-1)
    # decode steps on the post-prefill state, producing 1 new token per
    # (prompt, beam_step). Per-step semantics differ:
    #   * batched methods (paged, bs_kernel): n_steps == max_new - 1
    #   * sequential methods (tree, fasttree, mlca, adaptive_pool):
    #     n_steps == B * (max_new - 1) — each prompt accumulates separately.
    # The "per-token" metric normalizes by total tokens decoded (B*(max_new-1))
    # so it's apples-to-apples regardless of internal batching.
    total_tokens = B * (max_new - 1)
    per_token = decode_total / max(1, total_tokens)
    return Row(
        method=method_name,
        K=K, L_p=L_p, B=B, max_new=max_new,
        prefill_ms=timings["prefill_ms"],
        decode_total_ms=decode_total,
        decode_per_token_ms=decode_total / max(1, n_steps),
        decode_per_prompt_per_token_ms=per_token,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", nargs="+", type=int, default=[16, 32])
    ap.add_argument("--L_p", nargs="+", type=int, default=[8192])
    ap.add_argument("--B", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    methods = {k: METHODS[k] for k in args.methods if k in METHODS}
    if not methods:
        print(f"no valid methods in {args.methods}; valid: {list(METHODS.keys())}", file=sys.stderr)
        sys.exit(2)

    grid = [(K, L_p, B) for K in args.K for L_p in args.L_p for B in args.B]
    print(f"grid: {len(grid)} (K,L_p,B) cells × {len(methods)} methods = {len(grid) * len(methods)} runs")
    print(f"methods: {list(methods.keys())}")
    print(f"max_new: {args.max_new}")

    print("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    rows: list[Row] = []
    for (K, L_p, B) in grid:
        base = _make_prompt(tok, L_p)
        prompts = [list(base) for _ in range(B)]
        for name, fn in methods.items():
            print(f"  {name:<14} K={K:<3d} L_p={L_p:<6d} B={B:<2d}", end=" ", flush=True)
            t0 = time.perf_counter()
            row = run_one(name, fn, model, config, prompts, K, args.max_new)
            t1 = time.perf_counter()
            if row is not None:
                rows.append(row)
                print(
                    f"prefill={row.prefill_ms:7.1f}ms  "
                    f"decode_total={row.decode_total_ms:7.1f}ms  "
                    f"per_pt_token={row.decode_per_prompt_per_token_ms:6.2f}ms  "
                    f"({t1 - t0:5.1f}s wall)"
                )
            torch.cuda.empty_cache()

    out = args.out
    if out is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        gpu = torch.cuda.get_device_name().replace(" ", "_")
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = results_dir / f"bench_batched-{gpu}-{ts}.csv"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "K", "L_p", "B", "max_new",
            "prefill_ms", "decode_total_ms",
            "decode_per_token_ms", "decode_per_prompt_per_token_ms",
        ])
        for r in rows:
            w.writerow([
                r.method, r.K, r.L_p, r.B, r.max_new,
                f"{r.prefill_ms:.4f}",
                f"{r.decode_total_ms:.4f}",
                f"{r.decode_per_token_ms:.4f}",
                f"{r.decode_per_prompt_per_token_ms:.4f}",
            ])
    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

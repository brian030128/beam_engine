"""Sweep bs_kernel across its picker modes at L_p=8192.

Modes:
  1. picker            — let cost_model.pick_strategy_batch decide each step
  2. PER_BEAM          — force per-beam paged decode (no cascade)
  3. SHARED_2L_1POOL   — force 2-level cascade, single-pool ablation
                         (force_cta_tile_q=t_large)
  4. SHARED_2L_2POOL   — force 2-level cascade, default 2-pool routing
  5. SHARED_3L_1POOL   — force 3-level cascade, single-pool ablation
  6. SHARED_3L_2POOL   — force 3-level cascade, default 2-pool routing
  7. SHARED_2L_DEC_TAIL — force prefill-prefix + decode-tail + 1 merge
  8. SHARED_3L_DEC_TAIL — force prefill-prefix + prefill-inter + decode-tail + 2 merges

3-level picks fall back to 2-level when the workload has no uniform
intermediate groups (the common case for deterministic identical-prompt
batches), so SHARED_3L_* may not actually run as 3-level — flagged in
output.

Output CSV: per (mode, K, L_p, B) row with prefill_ms, decode_total_ms,
decode_per_prompt_per_token_ms.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from beam_engine.baselines.dbs import wrap as _dbs_wrap
from beam_engine.methods.bs_kernel import beam_search as bs_kernel_search
from beam_engine.methods.bs_kernel.cost_model import Strategy
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda"
DTYPE = torch.float16


# Mode name → set of allowed strategies (None = picker default).
MODES: dict[str, set[Strategy] | None] = {
    "picker":            None,
    "PER_BEAM":          {Strategy.PER_BEAM},
    "SHARED_2L_1POOL":   {Strategy.SHARED_2L_1POOL},
    "SHARED_2L_2POOL":   {Strategy.SHARED_2L_2POOL},
    "SHARED_3L_1POOL":   {Strategy.SHARED_3L_1POOL},
    "SHARED_3L_2POOL":   {Strategy.SHARED_3L_2POOL},
    "SHARED_2L_DEC_TAIL": {Strategy.SHARED_2L_DEC_TAIL},
    "SHARED_3L_DEC_TAIL": {Strategy.SHARED_3L_DEC_TAIL},
}


@dataclass
class Row:
    mode: str
    K: int
    L_p: int
    B: int
    max_new: int
    prefill_ms: float
    decode_total_ms: float
    decode_per_prompt_per_token_ms: float
    pick_hist: str = ""  # "d2_p1=16,d3_p1=240" — flat histogram of (depth, pool) picks across decode steps


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
    mode_name: str,
    available: set[Strategy] | None,
    model,
    config,
    prompts: list[list[int]],
    K: int,
    max_new: int,
    *,
    use_dbs: bool = False,
) -> Row | None:
    B = len(prompts)
    L_p = len(prompts[0])
    page_size = 16
    needed_pages = (
        sum((len(p) + page_size - 1) // page_size for p in prompts)
        + B * K * ((max_new + page_size) // page_size + 2)
        + 256
    )
    method_fn = (
        _dbs_wrap(bs_kernel_search) if use_dbs else bs_kernel_search
    )
    sig = inspect.signature(bs_kernel_search)
    extra: dict = {}
    if "max_num_pages" in sig.parameters:
        extra["max_num_pages"] = needed_pages
    if available is not None:
        extra["available_strategies"] = available

    try:
        beams, timings, picks = method_fn(
            model, config, prompts, max_new, K,
            return_timings=True, return_picks=True, **extra,
        )
    except Exception as e:
        print(f"  [{mode_name}] FAILED: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()
        return None

    decode_total = sum(timings["decode_step_ms"])
    # Picks: list per prompt of per-step Pick objects. All B prompts share
    # the same per-step pick (batched picker), so we read prompt 0.
    from collections import Counter
    hist = Counter()
    _DEC_TAIL = {Strategy.SHARED_2L_DEC_TAIL, Strategy.SHARED_3L_DEC_TAIL}
    for p in picks[0]:
        if p.strategy == Strategy.PER_BEAM:
            hist["per_beam"] += 1
        elif p.strategy in _DEC_TAIL:
            hist[f"d{p.depth}_dec_tail"] += 1
        else:
            hist[f"d{p.depth}_p{p.pool_count}"] += 1
    pick_hist = ",".join(f"{k}={v}" for k, v in sorted(hist.items()))
    return Row(
        mode=mode_name, K=K, L_p=L_p, B=B, max_new=max_new,
        prefill_ms=timings["prefill_ms"],
        decode_total_ms=decode_total,
        decode_per_prompt_per_token_ms=decode_total / max(1, B * (max_new - 1)),
        pick_hist=pick_hist,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", nargs="+", type=int, default=[16, 64])
    ap.add_argument("--L_p", nargs="+", type=int, default=[8192])
    ap.add_argument("--B", nargs="+", type=int, default=[1, 16, 32])
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    ap.add_argument("--dbs", action="store_true",
                    help="wrap bs_kernel with diverse beam search (G=4, λ=0.5)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bad = [m for m in args.modes if m not in MODES]
    if bad:
        print(f"unknown modes: {bad}; valid: {list(MODES.keys())}", file=sys.stderr)
        sys.exit(2)

    grid = [(K, L_p, B) for K in args.K for L_p in args.L_p for B in args.B]
    print(f"grid: {len(grid)} cells × {len(args.modes)} modes = {len(grid) * len(args.modes)} runs")
    print(f"modes: {args.modes}")

    print("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    rows: list[Row] = []
    for (K, L_p, B) in grid:
        base = _make_prompt(tok, L_p)
        prompts = [list(base) for _ in range(B)]
        for mode in args.modes:
            available = MODES[mode]
            label = f"dbs_{mode}" if args.dbs else mode
            print(
                f"  {label:<22} K={K:<3d} L_p={L_p:<6d} B={B:<2d}",
                end=" ", flush=True,
            )
            t0 = time.perf_counter()
            row = run_one(
                label, available, model, config, prompts, K, args.max_new,
                use_dbs=args.dbs,
            )
            t1 = time.perf_counter()
            if row is not None:
                rows.append(row)
                print(
                    f"prefill={row.prefill_ms:7.1f}ms  "
                    f"decode_total={row.decode_total_ms:8.1f}ms  "
                    f"per_pt_token={row.decode_per_prompt_per_token_ms:6.2f}ms  "
                    f"picks=[{row.pick_hist}]  "
                    f"({t1 - t0:5.1f}s wall)"
                )
            torch.cuda.empty_cache()

    out = args.out
    if out is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        gpu = torch.cuda.get_device_name().replace(" ", "_")
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = results_dir / f"bench_modes-{gpu}-{ts}.csv"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "mode", "K", "L_p", "B", "max_new",
            "prefill_ms", "decode_total_ms", "decode_per_prompt_per_token_ms",
            "pick_hist",
        ])
        for r in rows:
            w.writerow([
                r.mode, r.K, r.L_p, r.B, r.max_new,
                f"{r.prefill_ms:.4f}",
                f"{r.decode_total_ms:.4f}",
                f"{r.decode_per_prompt_per_token_ms:.4f}",
                r.pick_hist,
            ])
    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

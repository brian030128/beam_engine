"""Oracle-vs-model regret distribution for the bs_kernel cost model (B1).

For each workload point, force the bs_kernel driver into each available
strategy, measure end-to-end decode latency, and compare the cost
model's *predicted* best (its actual pick) against the *empirical* best
(the strategy with the lowest measured time — the "oracle").

Outputs:
  * per-workload regret = (model_pick_time / oracle_pick_time) - 1
  * distribution summary (mean, p50, p90, p99, max)
  * cells where model and oracle disagree (regret > 5%) → case study

Per CLAUDE.md: pick a fully-idle GPU; pin via CUDA_VISIBLE_DEVICES.

Usage:
    nvidia-smi
    CUDA_VISIBLE_DEVICES=<id> uv run python benchmarks/bs_kernel/oracle_vs_model.py [--full]
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from beam_engine.methods import bs_kernel
from beam_engine.methods.bs_kernel.cost_model import (
    Coefficients,
    Strategy,
    pick_strategy,
)
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16


# Strategies the harness probes per workload. PER_BEAM and SHARED 2-level
# variants are always candidates; SHARED 3-level variants are filtered
# at runtime (require intermediate-group structure that synthetic prompts
# rarely produce).
PROBE_STRATEGIES = [
    Strategy.PER_BEAM,
    Strategy.SHARED_2L_1POOL,
    Strategy.SHARED_2L_2POOL,
]

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


def _make_prompt(tokenizer, target_len: int) -> list[int]:
    snippet = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
        "She traveled across mountains and rivers, recording her findings "
        "in a leather-bound journal that grew thicker with every passing "
        "season. The townsfolk welcomed her with warm hearts and cool drinks. "
    )
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
    return s[max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))]


@dataclass
class WorkloadResult:
    K: int
    L_p: int
    max_new: int
    times_per_strategy: dict[str, float]   # median decode_step_ms
    model_pick: str
    oracle_pick: str
    regret: float


def run_strategy(model, config, prompt_ids, K, max_new, strategy: Strategy | None) -> float | None:
    """Return median decode_step_ms for the chosen forced-strategy run, or
    None if the strategy is unavailable for this workload.
    """
    available = {strategy} if strategy is not None else None
    page_size = 16
    needed_pages = (
        (len(prompt_ids) + page_size - 1) // page_size
        + K * ((max_new + page_size) // page_size)
        + 256
    )
    try:
        _, timings = bs_kernel.beam_search(
            model, config, [prompt_ids], max_new, K,
            return_timings=True,
            available_strategies=available,
            max_num_pages=needed_pages,
        )
    except ValueError as e:
        # cost_model raises when filtered candidate set is empty (e.g.
        # SHARED_3L_* without intermediate groups).
        return None
    except Exception as e:
        print(f"    forced={strategy.value if strategy else 'auto'} FAILED: {type(e).__name__}: {e}")
        return None

    decode_ms = timings["decode_step_ms"]
    if not decode_ms:
        return None
    return statistics.median(decode_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    grid = FULL_GRID if args.full else QUICK_GRID

    print("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print(f"grid: {len(grid)} workloads × {len(PROBE_STRATEGIES) + 1} runs each")

    results: list[WorkloadResult] = []
    page_size = 16
    for (K, L_p, max_new) in grid:
        prompt_ids = _make_prompt(tok, L_p)
        needed_pages = (
            (L_p + page_size - 1) // page_size
            + K * ((max_new + page_size) // page_size)
            + 256
        )
        print(f"\n=== K={K} L_p={L_p} max_new={max_new} ===")

        # Per-strategy probe.
        per_strategy: dict[str, float] = {}
        for s in PROBE_STRATEGIES:
            t = run_strategy(model, config, prompt_ids, K, max_new, s)
            if t is not None:
                per_strategy[s.value] = t
                print(f"  forced {s.value:<22} median decode = {t:6.2f} ms")

        # Cost-model auto pick.
        t_auto = run_strategy(model, config, prompt_ids, K, max_new, strategy=None)
        # Need the actual strategy the cost model picked; re-pick it with
        # the same WorkloadShape. Approximation: assume cost_model picks
        # consistently across the run; consult the first step's pick.
        # For exactness, use the in-driver pick log:
        beams, _, picks = bs_kernel.beam_search(
            model, config, [prompt_ids], 2, K,
            return_timings=True, return_picks=True,
            max_num_pages=needed_pages,
        )
        first_pick = picks[0][0].strategy.value if picks[0] else "unknown"

        # Oracle = strategy with min measured time.
        if not per_strategy:
            print("  no strategy probes succeeded; skipping")
            continue
        oracle = min(per_strategy, key=lambda k: per_strategy[k])
        oracle_t = per_strategy[oracle]
        # Use the model-pick's *forced* time as the model's effective time.
        # If model_pick wasn't probed (unlikely in PROBE_STRATEGIES), fall
        # back to t_auto.
        model_t = per_strategy.get(first_pick, t_auto if t_auto is not None else float("inf"))
        regret = (model_t / oracle_t) - 1.0
        print(f"  model picks {first_pick}  (t_model={model_t:.2f}ms)")
        print(f"  oracle      {oracle:<22} (t_oracle={oracle_t:.2f}ms)")
        print(f"  regret      {regret*100:+.1f}%")
        results.append(WorkloadResult(
            K=K, L_p=L_p, max_new=max_new,
            times_per_strategy=per_strategy,
            model_pick=first_pick,
            oracle_pick=oracle,
            regret=regret,
        ))

    # Summary.
    if results:
        regrets = [r.regret for r in results]
        print("\n" + "=" * 60)
        print("Regret distribution:")
        print(f"  mean  = {statistics.mean(regrets) * 100:+.2f}%")
        print(f"  p50   = {_percentile(regrets, 0.50) * 100:+.2f}%")
        print(f"  p90   = {_percentile(regrets, 0.90) * 100:+.2f}%")
        print(f"  p99   = {_percentile(regrets, 0.99) * 100:+.2f}%")
        print(f"  max   = {max(regrets) * 100:+.2f}%")
        bad = [r for r in results if r.regret > 0.05]
        if bad:
            print(f"\n{len(bad)} workloads with regret > 5%:")
            for r in bad:
                print(f"  K={r.K} L_p={r.L_p}: model={r.model_pick} oracle={r.oracle_pick} regret={r.regret*100:+.1f}%")

    # Write CSV.
    out = args.out
    if out is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        gpu = torch.cuda.get_device_name().replace(" ", "_")
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = results_dir / f"oracle-vs-model-{gpu}-{ts}.csv"
    out = Path(out)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "K", "L_p", "max_new", "model_pick", "oracle_pick", "regret_pct",
            *[f"t_{s.value}_ms" for s in PROBE_STRATEGIES],
        ])
        for r in results:
            w.writerow([
                r.K, r.L_p, r.max_new, r.model_pick, r.oracle_pick,
                f"{r.regret*100:.2f}",
                *[f"{r.times_per_strategy.get(s.value, float('nan')):.4f}" for s in PROBE_STRATEGIES],
            ])
    print(f"\nwrote {len(results)} rows to {out}")


if __name__ == "__main__":
    main()

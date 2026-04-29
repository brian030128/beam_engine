"""Benchmark sweeps comparing paged vs tree attention beam search.

Two experiments designed to verify the documented weaknesses:

1. **Prefix sweep** — vary L_p ∈ {64, 256, 1024, 4096}, fix max_new small.
   Hypothesis (paged's weakness): paged's per-step decode time grows roughly
   linearly with L_p, with slope proportional to beam width K, because the
   shared prefix is re-read once per beam every step. Tree should grow with
   L_p too but with a slope independent of K (prefix read once total).

2. **Generation sweep** — fix L_p small, run max_new long, look at per-step
   time as a function of step number.
   Hypothesis (tree's weakness): tree's per-step time grows with step number
   (its mask covers a kv buffer that grows by K each step). Paged's per-step
   time grows much more slowly (each beam's suffix grows by 1 per step).
"""

from __future__ import annotations

import gc
import statistics

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import paged, tree
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_prompt(target_len: int, tokenizer) -> list[int]:
    """Synthesize a prompt of approximately ``target_len`` tokens."""
    text = (
        "Once upon a time, in a faraway land, there lived a brave knight who "
        "fought dragons and rescued villages. The kingdom was ancient and full "
        "of wonder. Mountains rose to the sky and rivers carved deep valleys. "
    ) * 200
    ids = tokenizer.encode(text)
    return ids[:target_len]


def run_paged(model, config, prompt_ids, max_new, K, max_pages=512):
    _, t = paged.beam_search(
        model, config, [prompt_ids], max_new, K,
        max_num_pages=max_pages, return_timings=True,
    )
    return t


def run_tree(model, config, prompt_ids, max_new, K):
    _, t = tree.beam_search(
        model, config, [prompt_ids], max_new, K, return_timings=True,
    )
    return t


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def fmt_ms(x: float) -> str:
    return f"{x:7.2f}"


def summarize_steps(steps: list[float]) -> dict[str, float]:
    if not steps:
        return {"mean": 0.0, "median": 0.0, "first": 0.0, "last": 0.0, "n": 0}
    return {
        "mean": statistics.mean(steps),
        "median": statistics.median(steps),
        "first": steps[0],
        "last": steps[-1],
        "n": len(steps),
    }


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def sweep_prefix(model, config, tokenizer):
    print()
    print("=" * 78)
    print("SWEEP 1 — Prefix length (paged's hypothesized weakness)")
    print("Fixed: max_new=8.  Vary: L_p, K.  Metric: average decode-step time (ms).")
    print("=" * 78)

    prefix_lens = [64, 256, 1024, 4096]
    beam_widths = [4, 8]
    max_new = 8

    header = f"{'L_p':>6} {'K':>3} | {'paged mean':>11} {'tree mean':>10} | {'speedup':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for K in beam_widths:
        for L_p in prefix_lens:
            prompt_ids = make_prompt(L_p, tokenizer)
            actual_lp = len(prompt_ids)

            t_p = run_paged(model, config, prompt_ids, max_new, K)
            cleanup()
            t_t = run_tree(model, config, prompt_ids, max_new, K)
            cleanup()

            sp = summarize_steps(t_p["decode_step_ms"])
            st = summarize_steps(t_t["decode_step_ms"])
            speedup = sp["mean"] / st["mean"] if st["mean"] > 0 else float("inf")
            rows.append((actual_lp, K, sp["mean"], st["mean"], speedup))
            print(f"{actual_lp:>6} {K:>3} | {fmt_ms(sp['mean']):>11} {fmt_ms(st['mean']):>10} | "
                  f"{speedup:>7.2f}x")
    print()
    print("Verdict — paged's weakness:")
    by_K: dict[int, list] = {}
    for lp, K, p, t, sp in rows:
        by_K.setdefault(K, []).append((lp, p, t, sp))
    for K, vals in by_K.items():
        vals.sort()
        first = vals[0]
        last = vals[-1]
        paged_growth = last[1] / first[1]
        tree_growth = last[2] / first[2]
        print(f"  K={K}: across L_p {first[0]}→{last[0]} ({last[0] / first[0]:.0f}x): "
              f"paged decode time grew {paged_growth:.2f}x, tree grew {tree_growth:.2f}x")
        if paged_growth > tree_growth + 0.2:
            print(f"        → paged scales WORSE with prefix length ✓ (hypothesis confirmed)")
        else:
            print(f"        → paged did NOT scale worse than tree (hypothesis not confirmed)")


def sweep_generation(model, config, tokenizer):
    print()
    print("=" * 78)
    print("SWEEP 2 — Generation length (tree's hypothesized weakness)")
    print("Fixed: L_p=64.  Vary: step number within a single max_new=512 run.")
    print("Metric: per-step decode time at step buckets [0..63 / 64..127 / ...].")
    print("=" * 78)

    L_p = 64
    max_new = 512
    K = 8
    bucket = 64

    prompt_ids = make_prompt(L_p, tokenizer)
    print(f"  Prompt length: {len(prompt_ids)}, K={K}, max_new={max_new}\n")

    t_p = run_paged(model, config, prompt_ids, max_new, K, max_pages=512)
    cleanup()
    t_t = run_tree(model, config, prompt_ids, max_new, K)
    cleanup()

    p_steps = t_p["decode_step_ms"]
    t_steps = t_t["decode_step_ms"]

    print(f"  {'step bucket':>14} | {'paged ms':>9} {'tree ms':>9} | {'tree/paged':>10}")
    print("  " + "-" * 50)
    n_steps = max(len(p_steps), len(t_steps))
    growth_data = []
    for start in range(0, n_steps, bucket):
        end = min(start + bucket, n_steps)
        p_avg = statistics.mean(p_steps[start:end]) if p_steps[start:end] else 0.0
        t_avg = statistics.mean(t_steps[start:end]) if t_steps[start:end] else 0.0
        ratio = t_avg / p_avg if p_avg > 0 else float("inf")
        growth_data.append((start, end, p_avg, t_avg, ratio))
        print(f"  {f'[{start}..{end - 1}]':>14} | {fmt_ms(p_avg):>9} {fmt_ms(t_avg):>9} | "
              f"{ratio:>9.2f}x")

    print()
    print("Verdict — tree's weakness:")
    if len(growth_data) >= 2:
        first = growth_data[0]
        last = growth_data[-1]
        paged_growth = last[2] / first[2] if first[2] > 0 else 0
        tree_growth = last[3] / first[3] if first[3] > 0 else 0
        print(f"  steps {first[0]}–{first[1]-1}: paged={fmt_ms(first[2])}ms  tree={fmt_ms(first[3])}ms")
        print(f"  steps {last[0]}–{last[1]-1}: paged={fmt_ms(last[2])}ms  tree={fmt_ms(last[3])}ms")
        print(f"  paged per-step time grew {paged_growth:.2f}x")
        print(f"  tree  per-step time grew {tree_growth:.2f}x")
        if tree_growth > paged_growth + 0.2:
            print(f"  → tree scales WORSE with generation length ✓ (hypothesis confirmed)")
        else:
            print(f"  → tree did NOT scale worse than paged (hypothesis not confirmed)")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    # Warmup — kernels JIT, autotuning, etc.
    print("Warming up...")
    warmup_ids = make_prompt(64, tokenizer)
    run_paged(model, config, warmup_ids, 4, 4)
    cleanup()
    run_tree(model, config, warmup_ids, 4, 4)
    cleanup()
    print("Warmup done.\n")

    sweep_prefix(model, config, tokenizer)
    sweep_generation(model, config, tokenizer)


if __name__ == "__main__":
    main()

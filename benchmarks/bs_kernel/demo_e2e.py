"""End-to-end beam-search demo: long-prefix prompt, real text generation.

Picks a realistic long-prefix workload and runs beam search through every
method (paged, tree, fasttree, adaptive_pool, bs_kernel). For each method:

  * total wall-clock time for the full rollout
  * decode-step p50 / p99 / mean
  * best-beam generated text (decoded from token ids)
  * speedup vs paged baseline

This is the "actually apply our kernel" demonstration — same prompt, same
beam width, same K outputs, just different attention dispatchers.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import fasttree, mlca, paged, tree
from beam_engine.methods import adaptive_pool, bs_kernel
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16


METHODS = {
    "paged":         paged.beam_search,
    "tree":          tree.beam_search,
    "fasttree":      fasttree.beam_search,
    "mlca":          mlca.beam_search,
    "adaptive_pool": adaptive_pool.beam_search,
    "bs_kernel":     bs_kernel.beam_search,
}


@dataclass
class Result:
    method: str
    total_wall_s: float
    prefill_ms: float
    decode_p50_ms: float
    decode_p99_ms: float
    decode_mean_ms: float
    best_text: str
    best_score: float


def _make_long_prompt(tokenizer, target_len: int, kind: str = "fairy"):
    if kind == "fairy":
        snippet = (
            "Once upon a time, in a kingdom far away, there lived a curious "
            "scholar named Elara who studied the stars and the ways of the "
            "natural world. She traveled across mountains and rivers, "
            "recording her findings in a leather-bound journal that grew "
            "thicker with every passing season. The townsfolk welcomed her "
            "with warm hearts and cool drinks, sharing stories of "
            "lighthouses and lost ships and ancient maps drawn on cracked "
            "parchment. One evening, an old fisherman approached her with "
            "a strange request that would change the course of her life. "
        )
    else:
        snippet = (
            "The protocol for distributed consensus operates by passing "
            "messages between nodes in a network until a quorum agrees. "
            "Each node maintains a local replica of the state and uses "
            "vector clocks to track causal dependencies. When a leader is "
            "elected, all subsequent writes flow through it before being "
            "broadcast. Consistency is maintained via a two-phase commit. "
        )
    repeats = max(8, target_len // 50)
    while True:
        ids = tokenizer.encode(snippet * repeats, add_special_tokens=False)
        if len(ids) >= target_len:
            break
        repeats *= 2
    return ids[:target_len]


def run_method(
    name, fn, model, config, prompt_ids, K, max_new, tokenizer,
):
    import inspect

    page_size = 16
    needed_pages = (
        (len(prompt_ids) + page_size - 1) // page_size
        + K * ((max_new + page_size) // page_size)
        + 256
    )
    extra = {}
    sig = inspect.signature(fn)
    if "max_num_pages" in sig.parameters:
        extra["max_num_pages"] = needed_pages

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    beams, timings = fn(
        model, config, [prompt_ids], max_new, K,
        return_timings=True, **extra,
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    decode_ms = timings["decode_step_ms"]
    best = beams[0][0]
    best_text = tokenizer.decode(best.token_ids, skip_special_tokens=True)

    return Result(
        method=name,
        total_wall_s=t1 - t0,
        prefill_ms=timings.get("prefill_ms", 0.0),
        decode_p50_ms=statistics.median(decode_ms) if decode_ms else float("nan"),
        decode_p99_ms=sorted(decode_ms)[int(0.99 * (len(decode_ms) - 1))] if decode_ms else float("nan"),
        decode_mean_ms=statistics.mean(decode_ms) if decode_ms else float("nan"),
        best_text=best_text,
        best_score=best.cum_log_prob,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--L_p", type=int, default=8192)
    ap.add_argument("--max_new", type=int, default=32)
    ap.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    args = ap.parse_args()

    print(f"=== Beam-search end-to-end demo ===")
    print(f"K={args.K}  L_p={args.L_p}  max_new={args.max_new}")
    print(f"model={MODEL_NAME}\n")

    print("Loading model and tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    prompt_ids = _make_long_prompt(tok, args.L_p)
    prompt_text = tok.decode(prompt_ids, skip_special_tokens=True)
    print(f"prompt length = {len(prompt_ids)} tokens")
    print(f"prompt prefix = {prompt_text[:160]!r}...\n")

    methods = {k: v for k, v in METHODS.items() if k in args.methods}
    results: list[Result] = []
    for name, fn in methods.items():
        print(f"--- running {name} ---")
        try:
            r = run_method(name, fn, model, config, prompt_ids,
                           args.K, args.max_new, tok)
            results.append(r)
            print(f"  total_wall = {r.total_wall_s:.2f}s  "
                  f"prefill = {r.prefill_ms:.0f}ms  "
                  f"decode_p50 = {r.decode_p50_ms:.1f}ms")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    # Headline table.
    print("\n" + "=" * 100)
    print(f"{'method':16s}  {'total_s':>8s}  {'prefill_ms':>11s}  "
          f"{'p50_ms':>8s}  {'p99_ms':>8s}  {'mean_ms':>8s}  {'speedup_decode':>14s}")
    print("-" * 100)
    paged_p50 = next((r.decode_p50_ms for r in results if r.method == "paged"), None)
    for r in results:
        speedup = (paged_p50 / r.decode_p50_ms) if paged_p50 else float("nan")
        print(f"{r.method:16s}  {r.total_wall_s:8.2f}  "
              f"{r.prefill_ms:11.0f}  "
              f"{r.decode_p50_ms:8.1f}  {r.decode_p99_ms:8.1f}  "
              f"{r.decode_mean_ms:8.1f}  {speedup:13.2f}×")

    # Best-beam outputs.
    print("\n" + "=" * 100)
    print(f"Best beam from each method (token continuation, score=cum_log_prob):")
    for r in results:
        snippet = r.best_text[:200].replace("\n", " ")
        print(f"\n  [{r.method}] score={r.best_score:.3f}")
        print(f"    {snippet!r}{'...' if len(r.best_text) > 200 else ''}")


if __name__ == "__main__":
    main()

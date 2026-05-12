"""Sweep BS_KERNEL_DEC_TAIL_PREFIX_SPLIT_PAGES at K=16 cells where
bs_kernel loses to fasttree, plus a K=64 cell where bs_kernel wins (to
check for regressions).

Per cell, runs ``fasttree`` once (baseline) and ``bs_kernel`` once per
forced split-pages value, prints ``forward_mean_ms``. Goal: see whether
forcing more kv-splits in the DEC_TAIL prefix prefill closes the K=16
gap without regressing K=64.

Usage:
    uv run python benchmarks/bs_kernel/sweep_prefix_split.py
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from beam_engine.baselines import fasttree
from beam_engine.methods import bs_kernel
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")
DEVICE = "cuda"
DTYPE = torch.float16
PAGE_SIZE = 16
MAX_NEW = 256


# (K, L_p, B): tuned to the cells where bs_kernel loses big at K=16 and
# wins at K=64 (sweep_paper_merged-20260512-1759 std mode).
CELLS = [
    (16,  8192, 16),
    (16,  8192, 32),
    (16, 30000,  8),
    (16, 30000, 16),
    (64,  8192, 16),  # control: bs_kernel currently wins
]

# Default (None / "") + forced fixed_split_size in pages.
SPLIT_PAGES_OPTIONS: list[int | None] = [None, 16, 32, 64, 128, 256]


def _make_prompt(tok, target_len: int) -> list[int]:
    seed = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
    )
    ids: list[int] = []
    while len(ids) < target_len:
        ids.extend(tok.encode(seed, add_special_tokens=False))
    return ids[:target_len]


def _phase_mean(timings, key):
    xs = timings.get(key, [])
    return (sum(xs) / len(xs)) if xs else 0.0


def _run_one(name, fn, model, config, prompts, K, B, L_p, needed_pages):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        out = fn(
            model, config, prompts, MAX_NEW, K,
            max_num_pages=needed_pages,
            return_timings=True,
            return_phase_timings=True,
        )
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"status": "OOM", "decode_per_token_ms": 0.0,
                "forward_mean_ms": 0.0, "plan_mean_ms": 0.0,
                "wall_s": time.perf_counter() - t0}
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    timings = out[1]
    steps = timings["decode_step_ms"]
    n = max(1, len(steps))
    return {
        "status": "OK",
        "decode_per_token_ms": sum(steps) / n,
        "forward_mean_ms":     _phase_mean(timings, "forward_ms"),
        "plan_mean_ms":        _phase_mean(timings, "plan_ms"),
        "wall_s": wall,
    }


def main():
    out_dir = Path(
        f"benchmarks/bs_kernel/results/prefix_split-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "prefix_split.csv"

    print(f"Loading model {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device=DEVICE,
    )
    config = model.config
    print("Model loaded.\n")

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "K", "L_p", "B", "method", "split_pages",
            "decode_per_token_ms", "forward_mean_ms", "plan_mean_ms",
            "wall_s", "status",
        ])

        for K, L_p, B in CELLS:
            base = _make_prompt(tok, L_p)
            prompts = [list(base) for _ in range(B)]
            unique_pages = sum(
                (len(p) + PAGE_SIZE - 1) // PAGE_SIZE
                for p in {tuple(p) for p in prompts}
            )
            needed_pages = (
                unique_pages
                + B * K * ((MAX_NEW + PAGE_SIZE) // PAGE_SIZE + 2)
                + 256
            )
            print(f"\n=== K={K} L_p={L_p} B={B} (pages={needed_pages}) ===",
                  flush=True)

            # fasttree baseline.
            r = _run_one(
                "fasttree", fasttree.beam_search,
                model, config, prompts, K, B, L_p, needed_pages,
            )
            print(f"  fasttree              "
                  f"fwd={r['forward_mean_ms']:6.2f}  "
                  f"step={r['decode_per_token_ms']:6.2f} ms  "
                  f"wall={r['wall_s']:.1f}s  [{r['status']}]",
                  flush=True)
            w.writerow([K, L_p, B, "fasttree", "",
                        r["decode_per_token_ms"], r["forward_mean_ms"],
                        r["plan_mean_ms"], r["wall_s"], r["status"]])
            ft_fwd = r["forward_mean_ms"]

            # bs_kernel with each forced split.
            for sp in SPLIT_PAGES_OPTIONS:
                if sp is None:
                    os.environ.pop("BS_KERNEL_DEC_TAIL_PREFIX_SPLIT_PAGES", None)
                    tag = "default"
                else:
                    os.environ["BS_KERNEL_DEC_TAIL_PREFIX_SPLIT_PAGES"] = str(sp)
                    tag = f"{sp:>3d}p"
                r = _run_one(
                    "bs_kernel", bs_kernel.beam_search,
                    model, config, prompts, K, B, L_p, needed_pages,
                )
                delta = r["forward_mean_ms"] - ft_fwd
                print(f"  bs_kernel split={tag:>7s}  "
                      f"fwd={r['forward_mean_ms']:6.2f}  "
                      f"Δft={delta:+5.2f}  "
                      f"step={r['decode_per_token_ms']:6.2f} ms  "
                      f"wall={r['wall_s']:.1f}s  [{r['status']}]",
                      flush=True)
                w.writerow([K, L_p, B, "bs_kernel",
                            "" if sp is None else sp,
                            r["decode_per_token_ms"], r["forward_mean_ms"],
                            r["plan_mean_ms"], r["wall_s"], r["status"]])
                f.flush()

            os.environ.pop("BS_KERNEL_DEC_TAIL_PREFIX_SPLIT_PAGES", None)

    print(f"\nDone. Results: {csv_path}")


if __name__ == "__main__":
    main()

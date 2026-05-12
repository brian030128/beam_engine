"""Compare bs_kernel strategies head-to-head at K=16 cells where the
picker chose SHARED_2L_DEC_TAIL but loses to fasttree.

Hypothesis: at mid-K (K=16), per-beam tail is small (~256 tokens), so
DEC_TAIL's two-kernel + merge overhead may exceed the cost a fused
1POOL cascade pays for handling the tail directly. Force each candidate
strategy and measure forward_mean_ms.

Usage:
    uv run python benchmarks/bs_kernel/sweep_strategy_force.py
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
from beam_engine.methods.bs_kernel.cost_model import Strategy
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = os.environ.get("BE_MODEL", "meta-llama/Llama-3.2-1B")
DEVICE = "cuda"
DTYPE = torch.float16
PAGE_SIZE = 16
MAX_NEW = 256


CELLS = [
    (16,  8192, 16),
    (16,  8192, 32),
    (16, 30000,  8),
    (16, 30000, 16),
    (64,  8192, 16),  # control
]

# Each entry: (label, available_strategies). None = picker default.
STRATEGY_OPTIONS = [
    ("default",        None),
    ("2L_1POOL",       {Strategy.SHARED_2L_1POOL}),
    ("3L_1POOL",       {Strategy.SHARED_3L_1POOL}),
    ("2L_DEC_TAIL",    {Strategy.SHARED_2L_DEC_TAIL}),
    ("PER_BEAM",       {Strategy.PER_BEAM}),
]


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


def _run(name, fn, model, config, prompts, K, B, L_p, needed_pages,
         available_strategies):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    kwargs = dict(
        max_num_pages=needed_pages,
        return_timings=True,
        return_phase_timings=True,
    )
    if available_strategies is not None:
        kwargs["available_strategies"] = available_strategies
    try:
        out = fn(model, config, prompts, MAX_NEW, K, **kwargs)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"status": "OOM", "decode_per_token_ms": 0.0,
                "forward_mean_ms": 0.0, "plan_mean_ms": 0.0,
                "wall_s": time.perf_counter() - t0}
    except Exception as e:
        torch.cuda.empty_cache()
        return {"status": f"FAIL_{type(e).__name__}",
                "decode_per_token_ms": 0.0, "forward_mean_ms": 0.0,
                "plan_mean_ms": 0.0,
                "wall_s": time.perf_counter() - t0,
                "err": f"{type(e).__name__}: {e}"}
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
        f"benchmarks/bs_kernel/results/strategy_force-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "strategy_force.csv"

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
            "K", "L_p", "B", "method", "strategy_label",
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

            r = _run("fasttree", fasttree.beam_search, model, config,
                     prompts, K, B, L_p, needed_pages, None)
            print(f"  fasttree                  "
                  f"fwd={r['forward_mean_ms']:6.2f}  "
                  f"step={r['decode_per_token_ms']:6.2f} ms  "
                  f"wall={r['wall_s']:5.1f}s  [{r['status']}]",
                  flush=True)
            w.writerow([K, L_p, B, "fasttree", "",
                        r["decode_per_token_ms"], r["forward_mean_ms"],
                        r["plan_mean_ms"], r["wall_s"], r["status"]])
            ft_fwd = r["forward_mean_ms"]

            for label, strats in STRATEGY_OPTIONS:
                r = _run("bs_kernel", bs_kernel.beam_search, model, config,
                         prompts, K, B, L_p, needed_pages, strats)
                if r["status"] == "OK":
                    delta = r["forward_mean_ms"] - ft_fwd
                    line = (
                        f"  bs_kernel [{label:12s}]  "
                        f"fwd={r['forward_mean_ms']:6.2f}  "
                        f"Δft={delta:+6.2f}  "
                        f"step={r['decode_per_token_ms']:6.2f} ms  "
                        f"wall={r['wall_s']:5.1f}s  [{r['status']}]"
                    )
                else:
                    line = (
                        f"  bs_kernel [{label:12s}]  "
                        f"[{r['status']}] {r.get('err', '')}"
                    )
                print(line, flush=True)
                w.writerow([K, L_p, B, "bs_kernel", label,
                            r["decode_per_token_ms"], r["forward_mean_ms"],
                            r["plan_mean_ms"], r["wall_s"], r["status"]])
                f.flush()

    print(f"\nDone. Results: {csv_path}")


if __name__ == "__main__":
    main()

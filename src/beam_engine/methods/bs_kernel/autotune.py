"""Per-device cost-model auto-tuning.

The hardware-physics calibration in `calibrate.py` measures coefficients
the cost expressions need (HBM bandwidth, launch overhead, merge cost,
per-tile compute). But the closed-form expressions don't fully capture
every device-specific effect:

* L2-cache reuse for PER_BEAM at small K — at K=4 the same prefix
  pages are read 4× per step but mostly land in L2, so the model's
  HBM-bound bandwidth prediction over-penalizes PER_BEAM.
* SM-occupancy effects for `pool_count=2` — on small-SM-count parts
  (A6000 = 84 SMs) a single saturating launch already fills the GPU,
  so the second pool's CTAs are pure launch+sync overhead.

We absorb these effects via two free parameters on `Coefficients`:
``share_extra_us`` and ``dual_pool_extra_us`` (see cost_model.py for
docs). Auto-tune fits them per GPU by:

1. Running a small benchmark grid (`AUTOTUNE_GRID`, 9 cells × 3
   strategies) measuring actual decode_step_ms.
2. Grid-searching the parameter pair to minimize total regret across
   the grid (regret = model_pick_time / oracle_pick_time − 1).
3. Persisting the tuned values to the same JSON cache as the
   physics-calibrated coefficients.

This is the cost-model equivalent of FlashInfer's per-device kernel
JIT specialization — at the cost-model layer rather than the kernel
layer.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from .cost_model import (
    Coefficients,
    IntermediateShape,
    Strategy,
    WorkloadShape,
    pick_strategy,
)


# Calibration grid — covers the L_p > 1024 regime where strategy choice
# matters. Trimmed to keep wall time reasonable (~3-5 min on RTX-class
# GPUs).
AUTOTUNE_GRID = [
    (16, 2048), (16, 8192), (16, 32768), (16, 65536),
    (32, 2048), (32, 8192), (32, 32768), (32, 65536),
    (64, 2048), (64, 8192), (64, 32768), (64, 65536),
]

AUTOTUNE_STRATEGIES = [
    Strategy.PER_BEAM,
    Strategy.SHARED_2L_1POOL,
    Strategy.SHARED_2L_2POOL,
]

# 2-D grid for the parameter search. Coarse on purpose — finer grids
# don't help past calibration noise.
SHARE_EXTRA_GRID_US = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
DUAL_POOL_EXTRA_GRID_US = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def _make_workload(
    K: int,
    L_p: int,
    suffix_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> WorkloadShape:
    """Approximate WorkloadShape for a (K, L_p) cell mid-decode.

    Suffix length is set per the typical post-prefill state — beams
    have written ~suffix_len tokens since divergence. Intermediate is
    omitted (synthetic prompts rarely produce uniform fork groups).
    """
    return WorkloadShape(
        K=K,
        L_p=L_p,
        suffix_lens=[suffix_len] * K,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_kv=2 * num_kv_heads * head_dim * dtype_bytes,
        intermediate=None,
    )


def _measure_strategy_times(
    model,
    config,
    tokenizer,
    *,
    coefficients: Coefficients,
    grid: list[tuple[int, int]],
    max_new: int,
    device: str,
    dtype: torch.dtype,
) -> dict[tuple[int, int], dict[str, float]]:
    """For each cell × forced strategy, run beam_search and record the
    median decode_step_ms. Strategies that fail (e.g. degenerate
    1-level cascade at very short L_p) are silently skipped.
    """
    from beam_engine.methods import bs_kernel

    base = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar who studied the stars and the ways of the natural world. "
        "She traveled across mountains and rivers, recording her findings "
        "in a leather-bound journal that grew thicker with every passing "
        "season. The townsfolk welcomed her with warm hearts and cool drinks. "
    ) * 200
    base_ids = tokenizer.encode(base, add_special_tokens=False)

    times: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    page_size = 16
    for (K, L_p) in grid:
        prompt_ids = base_ids[:L_p]
        needed_pages = (
            (L_p + page_size - 1) // page_size
            + K * ((max_new + page_size) // page_size)
            + 256
        )
        for s in AUTOTUNE_STRATEGIES:
            try:
                _, timings = bs_kernel.beam_search(
                    model, config, [prompt_ids], max_new, K,
                    return_timings=True,
                    coefficients=coefficients,
                    available_strategies={s},
                    device=device,
                    dtype=dtype,
                    max_num_pages=needed_pages,
                )
            except Exception as e:
                print(f"  [autotune] K={K} L_p={L_p} {s.value} FAILED: "
                      f"{type(e).__name__}: {e}")
                continue
            decode_ms = timings.get("decode_step_ms", [])
            if not decode_ms:
                print(f"  [autotune] K={K} L_p={L_p} {s.value} no decode_step_ms")
                continue
            times[(K, L_p)][s.value] = statistics.median(decode_ms)
    return dict(times)


def _eval_regret(
    coeff: Coefficients,
    times: dict[tuple[int, int], dict[str, float]],
    *,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    suffix_len: int,
) -> tuple[float, float, int]:
    """Sum-of-regrets and worst-cell regret for a candidate Coefficients
    against the measured grid. Cells where the model picks a strategy
    that wasn't measured (or no strategies were measured) are skipped.
    """
    total = 0.0
    worst = 0.0
    counted = 0
    for (K, L_p), cell in times.items():
        if not cell:
            continue
        w = _make_workload(K, L_p, suffix_len, num_kv_heads, head_dim, dtype_bytes)
        pick = pick_strategy(w, coeff)
        if pick.strategy.value not in cell:
            # Modeled pick not measured (rare); skip.
            continue
        model_t = cell[pick.strategy.value]
        oracle_t = min(cell.values())
        regret = (model_t / oracle_t) - 1.0
        total += regret
        worst = max(worst, regret)
        counted += 1
    return total, worst, counted


def autotune(
    model,
    config,
    *,
    coefficients: Coefficients,
    tokenizer,
    grid: list[tuple[int, int]] = AUTOTUNE_GRID,
    max_new: int = 16,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    verbose: bool = False,
) -> Coefficients:
    """Auto-tune ``share_extra_us`` and ``dual_pool_extra_us`` to
    minimize total regret on the grid.

    Returns a new Coefficients with the tuned values filled in. The
    physics-calibrated B_hbm / launch_us / sync_us / merge_us are
    passed through unchanged.

    ``per_tile_us`` is **reset to defaults** before measurement: the
    paged-prefill probe in ``calibrate.py`` over-estimates per-tile
    compute by including bandwidth and treating tiles as serial; the
    cost expression then over-penalizes SHARED at high K, which
    flips the empirically-correct SHARED pick to PER_BEAM. Until the
    probe is rewritten to isolate compute (or we add SM-occupancy
    modeling), the closed-form scaling on default per_tile_us values
    plus the auto-tuned overheads gives the most accurate picks.

    Cost: each grid cell runs ``len(AUTOTUNE_STRATEGIES) = 3`` short
    beam_search calls. With the default 9-cell grid × max_new=16
    that's 27 calls × ~1-2 s each = 30-60 s wall time per device.
    """
    # Reset per_tile_us to defaults; keep physics-calibrated bandwidth /
    # launch / sync / merge coefficients.
    defaults = Coefficients.defaults()
    coefficients = replace(coefficients, per_tile_us=dict(defaults.per_tile_us))

    if verbose:
        print(f"[autotune] using defaults for per_tile_us={coefficients.per_tile_us}")
        print(f"[autotune] running {len(grid)} cells × {len(AUTOTUNE_STRATEGIES)} strategies")

    times = _measure_strategy_times(
        model, config, tokenizer,
        coefficients=coefficients,
        grid=grid,
        max_new=max_new,
        device=device,
        dtype=dtype,
    )
    if verbose:
        n_pairs = sum(len(c) for c in times.values())
        print(f"[autotune] measured {n_pairs} (cell, strategy) pairs:")
        for (K, L_p) in grid:
            cell = times.get((K, L_p), {})
            if not cell:
                print(f"  K={K:>3d} L_p={L_p:>5d}  (no data)")
                continue
            best_s = min(cell, key=lambda k: cell[k])
            row = "  ".join(f"{s}={cell[s]:6.2f}" for s in sorted(cell))
            print(f"  K={K:>3d} L_p={L_p:>5d}  {row}  oracle={best_s}")

    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    # suffix_len at mid-decode is small (a few tokens at most by max_new).
    # The cost expressions are dominated by L_p so this isn't sensitive.
    suffix_len = 1

    best_total = float("inf")
    best_share = 0.0
    best_dual = 0.0
    best_worst = float("inf")
    best_n = 0

    for share_extra in SHARE_EXTRA_GRID_US:
        for dual_pool_extra in DUAL_POOL_EXTRA_GRID_US:
            cand = replace(
                coefficients,
                share_extra_us=share_extra,
                dual_pool_extra_us=dual_pool_extra,
            )
            total, worst, n = _eval_regret(
                cand, times,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype_bytes=dtype_bytes,
                suffix_len=suffix_len,
            )
            if n == 0:
                continue
            # Tie-break on worst-case regret to prefer smoother predictors.
            if (total < best_total) or (
                total == best_total and worst < best_worst
            ):
                best_total = total
                best_worst = worst
                best_share = share_extra
                best_dual = dual_pool_extra
                best_n = n

    if verbose:
        avg = best_total / max(best_n, 1)
        print(
            f"[autotune] best share_extra_us={best_share} "
            f"dual_pool_extra_us={best_dual}  "
            f"avg regret={avg*100:+.2f}%  worst={best_worst*100:+.2f}% "
            f"({best_n} cells)"
        )

    return replace(
        coefficients,
        share_extra_us=best_share,
        dual_pool_extra_us=best_dual,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main():
    import argparse
    from transformers import AutoTokenizer

    from beam_engine.models.modeling_llama import LlamaForCausalLM
    from .calibrate import calibrate, _save, _cache_path

    ap = argparse.ArgumentParser(description="Auto-tune cost-model overheads.")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--save", action="store_true",
                    help="overwrite the calibration cache with the tuned values")
    args = ap.parse_args()

    print(f"[autotune] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = LlamaForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device=args.device,
    )
    config = model.config

    coeffs = calibrate(args.device, force=False, verbose=True)
    print(f"[autotune] starting from physics-calibrated coefficients")

    tuned = autotune(
        model, config,
        coefficients=coeffs,
        tokenizer=tok,
        max_new=args.max_new,
        device=args.device,
        verbose=True,
    )

    print(f"\n[autotune] tuned coefficients:")
    print(f"  share_extra_us     = {tuned.share_extra_us}")
    print(f"  dual_pool_extra_us = {tuned.dual_pool_extra_us}")

    if args.save:
        path = _cache_path(torch.device(args.device))
        _save(tuned, path)
        print(f"[autotune] wrote {path}")
    else:
        print("[autotune] not saving (pass --save to overwrite cache)")


if __name__ == "__main__":
    _main()

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
    pick_strategy_batch,
)


# Calibration grid — covers the L_p > 1024 regime where strategy choice
# matters. Includes B>1 cells so dual-pool overhead (which scales with
# total tile count = B × per-prompt tiles) gets fitted; pre-fix we only
# probed B=1 and dual_pool_extra_us defaulted to 0, which produced bad
# 3L_2POOL picks at B≥8.
AUTOTUNE_GRID: list[tuple[int, int, int]] = [
    # B=1 — classic single-prompt grid
    (16, 2048, 1), (16, 8192, 1), (16, 32768, 1),
    (32, 2048, 1), (32, 8192, 1), (32, 32768, 1),
    (64, 2048, 1), (64, 8192, 1), (64, 32768, 1),
    # B=8 — representative batched cells
    (16, 8192, 8), (32, 8192, 8), (64, 8192, 8),
    # B=32 — the cell where 3L_2POOL is mispicked when extras are 0
    (64, 8192, 32),
    # Mid-K large-B cells where DEC_TAIL was empirically mispicked on
    # H100 (sweep_paper_merged-20260512-1759, strategy_force-20260512-
    # 201006). Without these in the autotune grid, dec_tail_extra_us
    # has no signal to fit against and the picker keeps choosing
    # DEC_TAIL where 1POOL is 2× faster in fwd.
    (16,  8192, 16), (16,  8192, 32),
    (16, 32768,  8), (16, 32768, 16),
    # K=32/B=16 long-decode cell — the gov_report L_p=8192 cell where
    # the picker over-selects DEC_TAIL by ~1481/2047 steps. forced
    # SHARED_3L_1POOL is 3% faster (48.7s vs 50.3s) at max_new=2048,
    # indicating dec_tail_extra_us is under-fit when probe max_new=256
    # (suffix_len=128) but evaluated against max_new=2048 (suffix~1024).
    (32,  8192, 16),
]

# Probe modes: filter sets passed to bs_kernel.beam_search. depth=3
# entries include the 2-level counterpart as fallback because 3-level
# layouts only fire on decode steps where the beam tree produced an
# intermediate group structure (early decode steps never have one).
MODE_FILTERS: dict[str, set[Strategy]] = {
    "per_beam": {Strategy.PER_BEAM},
    "2l1p":     {Strategy.SHARED_2L_1POOL},
    "2l2p":     {Strategy.SHARED_2L_2POOL},
    "3l1p":     {Strategy.SHARED_3L_1POOL, Strategy.SHARED_2L_1POOL},
    "3l2p":     {Strategy.SHARED_3L_2POOL, Strategy.SHARED_2L_2POOL},
    # DEC_TAIL probes — depth=3 falls back to depth=2 on early decode
    # steps that lack an intermediate group structure, same pattern as
    # the 3l1p/3l2p fallbacks above.
    "2ldt":     {Strategy.SHARED_2L_DEC_TAIL},
    "3ldt":     {Strategy.SHARED_3L_DEC_TAIL, Strategy.SHARED_2L_DEC_TAIL},
}

# Map a picked Strategy → the mode label whose measurement is the
# correct comparison point in _eval_regret.
STRATEGY_TO_MODE: dict[Strategy, str] = {
    Strategy.PER_BEAM:           "per_beam",
    Strategy.SHARED_2L_1POOL:    "2l1p",
    Strategy.SHARED_2L_2POOL:    "2l2p",
    Strategy.SHARED_3L_1POOL:    "3l1p",
    Strategy.SHARED_3L_2POOL:    "3l2p",
    Strategy.SHARED_2L_DEC_TAIL: "2ldt",
    Strategy.SHARED_3L_DEC_TAIL: "3ldt",
}

# Search grid for the two free parameters. Widened on the dual-pool
# axis: empirically, on H100 at B=32/K=64 the 2-pool dispatch carries
# ~50-100 µs of unmodeled per-call overhead — the previous max of 20 µs
# was a hard ceiling that prevented the right value being fit.
SHARE_EXTRA_GRID_US = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
DUAL_POOL_EXTRA_GRID_US = (
    0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0,
)
# dec_tail_extra_us is multiplied by B (len(workloads)) at the call
# site, so the per-step penalty at B=16 is 16× the value picked here.
# Empirical gap at K=16/B=16 on H100 is ~5 ms/step ≈ 312 µs/layer ≈ a
# few hundred µs · per-prompt at the picker's per-step cost scale —
# grid spans up to 500 µs/prompt to cover that and beyond.
DEC_TAIL_EXTRA_GRID_US = (
    0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
)
# Per-(workload, mean-tail-page) DEC_TAIL slack, added on top of
# ``dec_tail_extra_us``. Picker-side this is mean_tail_pages × B × value;
# at K=32/B=16 with suffix_len=512 (max_new=1024 probe regime),
# mean_tail_pages = 32 → value=0.5 contributes 16×32×0.5=256 µs to
# DEC_TAIL cost. Grid spans 0–4 µs/page; with default 0 the model
# reduces to the pre-existing constant DEC_TAIL slack.
DEC_TAIL_PER_TAIL_PAGE_GRID_US = (
    0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0,
)


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
    grid: list[tuple[int, int, int]],
    max_new: int,
    device: str,
    dtype: torch.dtype,
) -> dict[tuple[int, int, int], dict[str, float]]:
    """For each cell × probe-mode, run beam_search with the mode's
    filter and record the median decode_step_ms. Probes that fail
    (e.g. degenerate 1-level cascade at very short L_p) are skipped.

    Cells with B>1 use B identical prompts; the cost-model picker is
    invoked in batched form so cross-prompt wave-occupancy effects show
    up in the measurement (the very effect that makes
    ``dual_pool_extra_us`` matter).
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

    times: dict[tuple[int, int, int], dict[str, float]] = defaultdict(dict)
    page_size = 16
    for cell in grid:
        K, L_p, B = cell
        prompt_ids = base_ids[:L_p]
        prompts = [list(prompt_ids) for _ in range(B)]
        needed_pages = (
            B * ((L_p + page_size - 1) // page_size)
            + B * K * ((max_new + page_size) // page_size + 2)
            + 256
        )
        for mode, filt in MODE_FILTERS.items():
            try:
                _, timings = bs_kernel.beam_search(
                    model, config, prompts, max_new, K,
                    return_timings=True,
                    coefficients=coefficients,
                    available_strategies=filt,
                    device=device,
                    dtype=dtype,
                    max_num_pages=needed_pages,
                )
            except Exception as e:
                print(f"  [autotune] K={K} L_p={L_p} B={B} {mode} FAILED: "
                      f"{type(e).__name__}: {e}")
                continue
            decode_ms = timings.get("decode_step_ms", [])
            if not decode_ms:
                print(f"  [autotune] K={K} L_p={L_p} B={B} {mode} no decode_step_ms")
                continue
            times[cell][mode] = statistics.median(decode_ms)
    return dict(times)


def _eval_regret(
    coeff: Coefficients,
    times: dict[tuple[int, int, int], dict[str, float]],
    *,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    suffix_len: int,
) -> tuple[float, float, int]:
    """Sum-of-regrets and worst-cell regret for a candidate Coefficients
    against the measured grid. Cells where the model picks a strategy
    whose mode wasn't measured (or no modes were measured) are skipped.

    For each (K, L_p, B) cell we duplicate the workload B times and call
    ``pick_strategy_batch`` so the picker sees the same cross-prompt
    tile-sum it would at runtime. The picked Strategy is mapped to the
    matching probe-mode label via ``STRATEGY_TO_MODE``.
    """
    total = 0.0
    worst = 0.0
    counted = 0
    for (K, L_p, B), cell in times.items():
        if not cell:
            continue
        w = _make_workload(K, L_p, suffix_len, num_kv_heads, head_dim, dtype_bytes)
        pick = pick_strategy_batch([w] * B, coeff)
        mode = STRATEGY_TO_MODE.get(pick.strategy)
        if mode is None or mode not in cell:
            # Modeled pick not measured (rare); skip.
            continue
        model_t = cell[mode]
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
    grid: list[tuple[int, int, int]] = AUTOTUNE_GRID,
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
        print(f"[autotune] running {len(grid)} cells × {len(MODE_FILTERS)} probe modes")

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
        print(f"[autotune] measured {n_pairs} (cell, mode) pairs:")
        for cell_key in grid:
            K, L_p, B = cell_key
            cell = times.get(cell_key, {})
            if not cell:
                print(f"  K={K:>3d} L_p={L_p:>5d} B={B:>3d}  (no data)")
                continue
            best_s = min(cell, key=lambda k: cell[k])
            row = "  ".join(f"{s}={cell[s]:6.2f}" for s in sorted(cell))
            print(f"  K={K:>3d} L_p={L_p:>5d} B={B:>3d}  {row}  oracle={best_s}")

    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    # The empirical decode_step_ms is averaged over ``max_new`` decode
    # steps where the per-beam tail grows from 0 to ~max_new tokens.
    # The picker, however, is queried once per fit candidate with a
    # single WorkloadShape, so suffix_len needs to be the *median* of
    # the empirical distribution — ~max_new/2 — not 1.
    #
    # The earlier ``suffix_len=1`` made the autotune predict 1POOL-vs-
    # DEC_TAIL based on a workload where 1POOL is structurally cheaper
    # (no per-beam tail cost), even though at the realistic suffix=128
    # the cost model flips to DEC_TAIL. With suffix=1 the fit converges
    # to a tiny dec_tail_extra_us that has no effect at runtime — see
    # 2026-05-12 verification (picks 165/255 DEC_TAIL at K=16/B=16).
    suffix_len = max(1, max_new // 2)

    best_total = float("inf")
    best_share = 0.0
    best_dual = 0.0
    best_dec_tail = 0.0
    best_dec_tail_per_page = 0.0
    best_worst = float("inf")
    best_n = 0

    for share_extra in SHARE_EXTRA_GRID_US:
        for dual_pool_extra in DUAL_POOL_EXTRA_GRID_US:
            for dec_tail_extra in DEC_TAIL_EXTRA_GRID_US:
                for dec_tail_per_page in DEC_TAIL_PER_TAIL_PAGE_GRID_US:
                    cand = replace(
                        coefficients,
                        share_extra_us=share_extra,
                        dual_pool_extra_us=dual_pool_extra,
                        dec_tail_extra_us=dec_tail_extra,
                        dec_tail_per_tail_page_us=dec_tail_per_page,
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
                        best_dec_tail = dec_tail_extra
                        best_dec_tail_per_page = dec_tail_per_page
                        best_n = n

    if verbose:
        avg = best_total / max(best_n, 1)
        print(
            f"[autotune] best share_extra_us={best_share} "
            f"dual_pool_extra_us={best_dual} "
            f"dec_tail_extra_us={best_dec_tail} "
            f"dec_tail_per_tail_page_us={best_dec_tail_per_page}  "
            f"avg regret={avg*100:+.2f}%  worst={best_worst*100:+.2f}% "
            f"({best_n} cells)"
        )

    return replace(
        coefficients,
        share_extra_us=best_share,
        dual_pool_extra_us=best_dual,
        dec_tail_extra_us=best_dec_tail,
        dec_tail_per_tail_page_us=best_dec_tail_per_page,
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
    print(f"  share_extra_us            = {tuned.share_extra_us}")
    print(f"  dual_pool_extra_us        = {tuned.dual_pool_extra_us}")
    print(f"  dec_tail_extra_us         = {tuned.dec_tail_extra_us}")
    print(f"  dec_tail_per_tail_page_us = {tuned.dec_tail_per_tail_page_us}")

    if args.save:
        path = _cache_path(torch.device(args.device))
        _save(tuned, path)
        print(f"[autotune] wrote {path}")
    else:
        print("[autotune] not saving (pass --save to overwrite cache)")


if __name__ == "__main__":
    _main()

"""Cost-model grid harness.

After removal of the autotuned slack knobs (``share_extra_us`` and
``dec_tail_extra_us``), the picker has no free parameters to fit — the
cost expressions are pure closed-form over physics-calibrated
coefficients. This module is retained as a *regret reporting* tool
and as the source of the benchmark grid + measurement helpers used by
``benchmarks/bs_kernel/ablate_autotune_knobs.py``.

The CLI runs `AUTOTUNE_GRID` × `MODE_FILTERS`, measures decode_step_ms
for each (cell, mode), and prints picker regret of the current
coefficient cache against the per-cell oracle.
"""

from __future__ import annotations

import os
import statistics
from collections import defaultdict
from dataclasses import replace

import torch

from .cost_model import (
    Coefficients,
    DEFAULT_KV_DTYPE,
    IntermediateShape,
    Strategy,
    WorkloadShape,
    canonical_kv_dtype,
    pick_strategy,
    pick_strategy_batch,
)


# Calibration grid — covers the L_p > 1024 regime where strategy choice
# matters. Includes B>1 cells so batched cross-prompt occupancy effects
# are reflected in the per-cell measurements.
AUTOTUNE_GRID: list[tuple[int, int, int]] = [
    # B=1 — classic single-prompt grid
    (16, 2048, 1), (16, 8192, 1), (16, 32768, 1),
    (32, 2048, 1), (32, 8192, 1), (32, 32768, 1),
    (64, 2048, 1), (64, 8192, 1), (64, 32768, 1),
    # B=8 — representative batched cells
    (16, 8192, 8), (32, 8192, 8), (64, 8192, 8),
    # B=32 — large-batch cell
    (64, 8192, 32),
    # Mid-K large-B cells (DEC_TAIL boundary regimes on H100)
    (16,  8192, 16), (16,  8192, 32),
    (16, 32768,  8), (16, 32768, 16),
    # K=32/B=16 long-decode cell
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

def _make_workload(
    K: int,
    L_p: int,
    suffix_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    num_qo_heads: int = 0,
    kv_dtype: str = DEFAULT_KV_DTYPE,
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
        num_qo_heads=num_qo_heads,
        kv_dtype=kv_dtype,
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
    up in the measurement.
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
    num_qo_heads: int = 0,
    kv_dtype: str = DEFAULT_KV_DTYPE,
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
        w = _make_workload(K, L_p, suffix_len, num_kv_heads, head_dim, dtype_bytes,
                           num_qo_heads=num_qo_heads, kv_dtype=kv_dtype)
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


# ---------------------------------------------------------------------------
# CLI entry point — regret report on the autotune grid
# ---------------------------------------------------------------------------


def _main():
    import argparse
    from transformers import AutoTokenizer

    from beam_engine.models import load_model_for_causal_lm
    from beam_engine.distributed import (
        destroy_tp,
        get_tp_rank,
        get_tp_world_size,
        init_tp,
    )
    from .calibrate import calibrate

    ap = argparse.ArgumentParser(
        description="Report cost-model picker regret on AUTOTUNE_GRID."
    )
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument(
        "--dtype",
        choices=("fp16", "bf16"),
        default=os.environ.get("BE_DTYPE", "fp16"),
    )
    args = ap.parse_args()

    init_tp()
    tp_size = get_tp_world_size()
    tp_rank = get_tp_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = args.device if tp_size == 1 else f"cuda:{local_rank}"
    is_main = tp_rank == 0

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    if is_main:
        print(f"[autotune] tp_size={tp_size} device={device} dtype={dtype}")
        print(f"[autotune] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model_for_causal_lm(args.model, dtype=dtype, device=device)
    config = model.config

    coeffs = calibrate(device, force=False, verbose=is_main)
    # Reset per_tile_us to defaults (the calibration probe is
    # overhead-dominated — see measure_per_tile_us docstring).
    defaults = Coefficients.defaults()
    coeffs = replace(coeffs, per_tile_us=dict(defaults.per_tile_us))

    times = _measure_strategy_times(
        model, config, tok,
        coefficients=coeffs,
        grid=AUTOTUNE_GRID,
        max_new=args.max_new,
        device=device,
        dtype=dtype,
    )

    if not is_main:
        destroy_tp()
        return

    num_kv_heads = config.num_key_value_heads
    num_qo_heads = getattr(config, "num_attention_heads", num_kv_heads)
    head_dim = config.head_dim
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    suffix_len = max(1, args.max_new // 2)
    kv_dtype_label = canonical_kv_dtype(dtype)

    total, worst, n = _eval_regret(
        coeffs, times,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
        suffix_len=suffix_len,
        num_qo_heads=num_qo_heads,
        kv_dtype=kv_dtype_label,
    )
    print(
        f"\n[autotune] picker regret on grid: "
        f"total={total*100:+.2f}%  "
        f"avg={(total/max(n,1))*100:+.2f}%  "
        f"worst={worst*100:+.2f}%  "
        f"({n} cells)"
    )

    destroy_tp()


if __name__ == "__main__":
    _main()

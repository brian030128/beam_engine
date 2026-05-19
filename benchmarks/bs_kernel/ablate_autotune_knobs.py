"""Picker-regret report on the AUTOTUNE_GRID.

After removal of the autotuned slack knobs (``share_extra_us`` and
``dec_tail_extra_us``), the cost model has no free parameters to ablate
— this script now serves as a regression test: measure each cell ×
probe-mode, then report per-cell picker regret against the empirical
oracle.

Outputs:
  - Total / average / worst-cell regret (summary line).
  - Per-cell table: oracle strategy, picker strategy, regret %.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

import torch
from transformers import AutoTokenizer

from beam_engine.distributed import destroy_tp, get_tp_rank, get_tp_world_size, init_tp
from beam_engine.methods.bs_kernel.autotune import (
    AUTOTUNE_GRID,
    STRATEGY_TO_MODE,
    _eval_regret,
    _make_workload,
    _measure_strategy_times,
)
from beam_engine.methods.bs_kernel.calibrate import calibrate
from beam_engine.methods.bs_kernel.cost_model import (
    Coefficients,
    pick_strategy_batch,
)
from beam_engine.models import load_model_for_causal_lm


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    args = ap.parse_args()

    init_tp()
    tp_size = get_tp_world_size()
    tp_rank = get_tp_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = args.device if tp_size == 1 else f"cuda:{local_rank}"
    is_main = tp_rank == 0

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    if is_main:
        print(f"[ablate] tp_size={tp_size} device={device} dtype={dtype}")
        print(f"[ablate] model={args.model}  max_new={args.max_new}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model_for_causal_lm(args.model, dtype=dtype, device=device)
    config = model.config

    coeffs = calibrate(device, force=False, verbose=is_main)
    # Reset per_tile_us to defaults — the calibration probe is
    # overhead-dominated; the cost model uses the hardcoded defaults.
    defaults = Coefficients.defaults()
    coeffs = replace(coeffs, per_tile_us=dict(defaults.per_tile_us))

    if is_main:
        print(
            f"[ablate] base coefficients: B_hbm={coeffs.B_hbm:.0f} "
            f"per_tile_us={coeffs.per_tile_us} num_sms={coeffs.num_sms}"
        )
        print(f"[ablate] measuring grid: {len(AUTOTUNE_GRID)} cells")

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

    n_pairs = sum(len(c) for c in times.values())
    print(f"[ablate] measured {n_pairs} (cell, mode) pairs")
    print()

    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    suffix_len = max(1, args.max_new // 2)

    total, worst, n_cells = _eval_regret(
        coeffs, times,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
        suffix_len=suffix_len,
    )
    avg = total / max(n_cells, 1)
    print("=" * 80)
    print(
        f"picker regret on grid: total={_fmt_pct(total)}  "
        f"avg={_fmt_pct(avg)}  worst={_fmt_pct(worst)}  "
        f"({n_cells} cells)"
    )
    print("=" * 80)
    print()

    print(f"{'cell':22s} {'oracle':12s} {'pick':22s} {'regret':>8s}")
    print("-" * 70)
    for (K, L_p, B), cell in sorted(times.items()):
        if not cell:
            print(f"K={K:>3d} L={L_p:>5d} B={B:>3d}    (no data)")
            continue
        oracle = min(cell, key=lambda k: cell[k])
        oracle_t = cell[oracle]
        w = _make_workload(K, L_p, suffix_len, num_kv_heads, head_dim, dtype_bytes)
        pick = pick_strategy_batch([w] * B, coeffs)
        mode = STRATEGY_TO_MODE.get(pick.strategy)
        if mode is None or mode not in cell:
            reg_str = "(unmeasured)"
        else:
            model_t = cell[mode]
            reg = (model_t / oracle_t) - 1.0
            reg_str = _fmt_pct(reg)
        print(
            f"K={K:>3d} L={L_p:>5d} B={B:>3d}    "
            f"{oracle:12s} {pick.strategy.name:22s} {reg_str:>8s}"
        )

    destroy_tp()


if __name__ == "__main__":
    main()

"""Collate exp1c_v2 results into a per-model summary table.

For each (model, scenario) pulls:
  - bs_kernel decode_total_ms (the picker, free choice)
  - bsk_2l_1p / bs_kernel_2l1p decode_total_ms (forced SHARED_2L_1POOL)
  - bsk_2l_dt / bs_kernel_2ldt decode_total_ms (forced SHARED_2L_DEC_TAIL)
  - picker strategy mix (parsed from run.log "strategy mix: {...}" lines)

Prints a markdown table per model.

Usage:
    python scripts/paper-exp/collate_exp1c_v2.py
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "benchmarks/bs_kernel/results/paper-exp/exp1c_v2"

SCENARIOS = [
    ("multi_level_system",    "multi_level_system.csv"),
    ("multi_chain_reasoning", "multi_chain_reasoning.csv"),
    ("multi_few_shot",        "multi_few_shot.csv"),
    ("beam_search",           "beam_search.csv"),
]

# Method aliases differ between bench_sglang_e2e.py (sglang) and bench_batched.py.
PICKER_ALIASES = {"bs_kernel"}
FUSED_ALIASES = {"bsk_2l_1p", "bs_kernel_2l1p"}
DECTAIL_ALIASES = {"bsk_2l_dt", "bs_kernel_2ldt"}

# Regex for "strategy mix: {'SHARED_2L_1POOL': 255, ...}" lines in run.log.
MIX_LINE = re.compile(r"strategy mix:\s*(\{[^}]+\})")
CELL_HEADER = re.compile(r"=== cell:\s*(\S+)")


def _decode_total_ms(csv_path: Path) -> dict[str, float]:
    """Return {method: decode_total_ms} from a bench CSV.

    Both bench_sglang_e2e.py and bench_batched.py emit a 'method' and a
    'decode_total_ms' column; the rest of the columns differ. If a method
    appears more than once (multiple repeats), take the median.
    """
    if not csv_path.exists():
        return {}
    by_method: dict[str, list[float]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row.get("method", "")
            try:
                total = float(row.get("decode_total_ms", ""))
            except ValueError:
                continue
            by_method.setdefault(m, []).append(total)
    return {
        m: sorted(xs)[len(xs) // 2] for m, xs in by_method.items()
    }


def _parse_strategy_mixes(log_path: Path) -> dict[str, list[dict[str, int]]]:
    """Walk run.log and group "strategy mix" lines by the cell they
    belong to. Returns {cell_name: [mix_dict, ...]} preserving order.
    """
    if not log_path.exists():
        return {}
    out: dict[str, list[dict[str, int]]] = {}
    cur_cell: str | None = None
    for line in log_path.read_text().splitlines():
        m = CELL_HEADER.search(line)
        if m:
            cur_cell = m.group(1)
            out.setdefault(cur_cell, [])
            continue
        m = MIX_LINE.search(line)
        if m and cur_cell:
            try:
                d = ast.literal_eval(m.group(1))
                if isinstance(d, dict):
                    out[cur_cell].append({str(k): int(v) for k, v in d.items()})
            except (ValueError, SyntaxError):
                pass
    return out


def _picker_mix_for_cell(
    cell_mixes: list[dict[str, int]],
) -> dict[str, int]:
    """Each method run dumps 1 or 2 "strategy mix" lines (DEC_TAIL path
    + SHARED path; pure runs emit only one). All mix lines from a
    single run sum to the same total (= n_decode_steps for that cell).
    Group consecutive mixes greedily by accumulating until the running
    sum equals the cell's per-run total, then return the first run's
    merged mix (the picker's, since runs go picker → forced FUSED →
    forced DECTAIL in the sbatch).
    """
    if not cell_mixes:
        return {}
    # Per-run total = max single-mix sum (at least one forced run is
    # pure, so its only mix line sums to the full run total).
    target = max(sum(d.values()) for d in cell_mixes)
    runs: list[dict[str, int]] = []
    cur: dict[str, int] = {}
    cur_sum = 0
    for d in cell_mixes:
        for k, v in d.items():
            cur[k] = cur.get(k, 0) + v
        cur_sum += sum(d.values())
        if cur_sum >= target:
            runs.append(cur)
            cur = {}
            cur_sum = 0
    if cur:
        runs.append(cur)
    return runs[0] if runs else {}


def _fmt_mix(mix: dict[str, int]) -> str:
    if not mix:
        return "—"
    total = sum(mix.values())
    parts = []
    for k in sorted(mix, key=lambda s: -mix[s]):
        v = mix[k]
        parts.append(f"{k}×{v} ({100*v/total:.0f}%)")
    return ", ".join(parts)


def collate_model(model_dir: Path) -> str:
    tag = model_dir.name
    mixes = _parse_strategy_mixes(model_dir / "run.log")
    rows: list[tuple[str, str, str, str, str, str]] = []
    for scenario, csv_name in SCENARIOS:
        csv_path = model_dir / csv_name
        totals = _decode_total_ms(csv_path)

        def _pick(aliases: set[str]) -> str:
            for a in aliases:
                if a in totals:
                    return f"{totals[a]:.0f}"
            return "—"

        picker_t = _pick(PICKER_ALIASES)
        fused_t = _pick(FUSED_ALIASES)
        dectail_t = _pick(DECTAIL_ALIASES)
        # delta vs the better forced
        delta = "—"
        try:
            pf = float(picker_t)
            best = min(float(x) for x in (fused_t, dectail_t) if x != "—")
            delta = f"{(pf / best - 1.0) * 100:+.1f}%"
        except ValueError:
            pass
        picker_mix = _picker_mix_for_cell(mixes.get(scenario, []))
        rows.append((scenario, picker_t, fused_t, dectail_t, delta, _fmt_mix(picker_mix)))

    out = [f"### {tag}\n"]
    out.append("| scenario | picker (ms) | forced FUSED (ms) | forced DEC_TAIL (ms) | picker vs better forced | picker strategy mix |")
    out.append("|---|---:|---:|---:|---:|---|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    if not RESULTS_ROOT.exists():
        print(f"results dir not found: {RESULTS_ROOT}")
        return
    for d in sorted(RESULTS_ROOT.iterdir()):
        if d.is_dir():
            print(collate_model(d))


if __name__ == "__main__":
    main()

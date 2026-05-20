"""Tiny one-shot collator for the smallBK probe.

Reads benchmarks/bs_kernel/results/paper-exp/exp1c_v2_smallBK/<model>/
*.csv and prints a per-cell picker vs forced FUSED vs forced DEC_TAIL
table, plus the picker's strategy mix parsed from run.log.
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "benchmarks/bs_kernel/results/paper-exp"
CELL_HEADER = re.compile(r"=== cell:\s*(\S+)\s*\(B=(\d+),\s*K=(\d+)")
MIX_LINE = re.compile(r"strategy mix:\s*(\{[^}]+\})")


def parse_mixes(log_path: Path):
    out: dict[tuple[str, int, int], dict[str, int]] = {}
    cur = None
    pool: list[dict[str, int]] = []
    if not log_path.exists():
        return out

    def _flush(cur, pool):
        if cur is None or not pool:
            return
        # Per-method-run total = max single mix sum (forced runs are pure).
        target = max(sum(d.values()) for d in pool)
        cur_d: dict[str, int] = {}
        cur_sum = 0
        first_run = None
        for d in pool:
            for k, v in d.items():
                cur_d[k] = cur_d.get(k, 0) + v
            cur_sum += sum(d.values())
            if cur_sum >= target:
                if first_run is None:
                    first_run = cur_d
                cur_d = {}
                cur_sum = 0
        if first_run is not None:
            out[cur] = first_run

    for line in log_path.read_text().splitlines():
        m = CELL_HEADER.search(line)
        if m:
            _flush(cur, pool)
            cur = (m.group(1), int(m.group(2)), int(m.group(3)))
            pool = []
            continue
        m = MIX_LINE.search(line)
        if m and cur:
            try:
                d = ast.literal_eval(m.group(1))
                if isinstance(d, dict):
                    pool.append({str(k): int(v) for k, v in d.items()})
            except Exception:
                pass
    _flush(cur, pool)
    return out


def parse_csv(csv_path: Path):
    if not csv_path.exists():
        return {}
    by_method: dict[str, float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            try:
                by_method[row["method"]] = float(row["decode_total_ms"])
            except (KeyError, ValueError):
                continue
    return by_method


def fmt_mix(mix):
    if not mix:
        return "—"
    total = sum(mix.values())
    return ", ".join(
        f"{k}×{v}({100*v/total:.0f}%)" for k, v in sorted(mix.items(), key=lambda kv: -kv[1])
    )


def collate(probe_dir: Path):
    log = probe_dir / "run.log"
    mixes = parse_mixes(log)
    rows = []
    for csv_file in sorted(probe_dir.glob("*.csv")):
        name = csv_file.stem  # e.g. multi_chain_reasoning_B1_K16
        m = re.match(r"(.+)_B(\d+)_K(\d+)", name)
        if not m:
            continue
        scenario, B, K = m.group(1), int(m.group(2)), int(m.group(3))
        totals = parse_csv(csv_file)
        picker = totals.get("bs_kernel")
        fused = totals.get("bsk_2l_1p") or totals.get("bs_kernel_2l1p")
        dect = totals.get("bsk_2l_dt") or totals.get("bs_kernel_2ldt")
        better = None
        if fused is not None and dect is not None:
            better = "FUSED" if fused < dect else "DEC_TAIL"
            gap = (max(fused, dect) / min(fused, dect) - 1) * 100
        else:
            gap = None
        mix = mixes.get((scenario, B, K), {})
        rows.append((scenario, B, K, picker, fused, dect, better, gap, mix))

    # Sort: scenario then B then K
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    print(f"## {probe_dir.parent.name} / {probe_dir.name}\n")
    print("| scenario | B | K | picker (ms) | forced FUSED (ms) | forced DEC_TAIL (ms) | better forced | gap | picker mix |")
    print("|---|---:|---:|---:|---:|---:|---|---:|---|")
    for scenario, B, K, picker, fused, dect, better, gap, mix in rows:
        def fmt(x):
            return f"{x:.0f}" if isinstance(x, (int, float)) else "—"
        gap_s = f"{gap:.1f}%" if gap is not None else "—"
        print(f"| {scenario} | {B} | {K} | {fmt(picker)} | {fmt(fused)} | {fmt(dect)} | {better or '—'} | {gap_s} | {fmt_mix(mix)} |")


def main():
    for probe_root_name in ("exp1c_v2_smallBK", "exp1c_v2_longLp"):
        root = ROOT / probe_root_name
        if not root.exists():
            continue
        for model_dir in sorted(root.iterdir()):
            if model_dir.is_dir():
                collate(model_dir)
                print()


if __name__ == "__main__":
    main()

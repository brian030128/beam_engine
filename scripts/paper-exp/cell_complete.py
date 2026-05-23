"""Exit 0 if a result CSV holds a finite decode time for every expected
method, else exit 1. Used by the final_paper sbatch ladders to decide
whether a (K, B) attempt fully fit (all methods ran) or needs to shrink.

Usage:
    python cell_complete.py <csv> <kind:sglang|batched> <m1,m2,...>
"""
from __future__ import annotations

import csv
import math
import sys


def _finite_pos(s: str) -> bool:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: cell_complete.py <csv> <sglang|batched> <m1,m2,...>",
              file=sys.stderr)
        return 2
    path, kind, methods_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    expected = [m for m in methods_csv.split(",") if m]

    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        return 1
    if not rows:
        return 1

    col = "decode_total_ms"  # present in both sglang and batched output
    seen: set[str] = set()
    for r in rows:
        if _finite_pos(r.get(col, "")):
            seen.add(r.get("method", ""))

    missing = [m for m in expected if m not in seen]
    if missing:
        print(f"incomplete: missing {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Merge fresh bs_kernel rows into an existing picker_demo CSV.

After the cost model changed we want to refresh only the bs_kernel
methods (bs_kernel / bsk_2l_1p / bsk_2l_dt) on cells where the baseline
methods (paged / fasttree / deft) are already populated. This script
takes a fresh CSV that contains only the bs_kernel methods, drops any
old bs_kernel rows from the existing CSV, and stitches the new ones in.

Method aliases used by bench_batched.py and bench_sglang_e2e.py differ
slightly (``bs_kernel_2l1p`` vs ``bsk_2l_1p``) — both forms are treated
as bs_kernel methods for filtering.

Usage:
    python merge_picker_csv.py <existing.csv> <fresh.csv>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

BSK_METHODS = {
    "bs_kernel",
    "bsk_2l_1p", "bs_kernel_2l1p",
    "bsk_2l_dt", "bs_kernel_2ldt",
}


def _method_col(fieldnames: list[str]) -> str:
    if "method" in fieldnames:
        return "method"
    # bench_batched.py uses "method" as col 1; bench_sglang_e2e.py
    # uses "method" as col 3. Both name it "method", so this is
    # mostly defensive.
    raise SystemExit(f"no 'method' column in fields: {fieldnames}")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    existing = Path(sys.argv[1])
    fresh = Path(sys.argv[2])

    if not fresh.exists():
        raise SystemExit(f"fresh CSV missing: {fresh}")

    with fresh.open() as f:
        reader = csv.DictReader(f)
        fresh_header = reader.fieldnames or []
        fresh_rows = list(reader)

    method_col = _method_col(fresh_header)

    baseline_rows: list[dict] = []
    if existing.exists():
        with existing.open() as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames or []
            for row in reader:
                if row.get(method_col, "") in BSK_METHODS:
                    continue
                baseline_rows.append(row)
        # Use existing header if it has more columns; otherwise prefer
        # fresh header so new columns aren't dropped.
        header = (
            existing_header
            if len(existing_header) >= len(fresh_header)
            else fresh_header
        )
    else:
        header = fresh_header

    with existing.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in baseline_rows:
            writer.writerow({k: row.get(k, "") for k in header})
        for row in fresh_rows:
            writer.writerow({k: row.get(k, "") for k in header})

    print(
        f"merged: {existing.name} ← "
        f"{len(baseline_rows)} baselines + {len(fresh_rows)} bs_kernel rows"
    )


if __name__ == "__main__":
    main()

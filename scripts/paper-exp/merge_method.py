"""Merge one method's rows from a temp CSV into a canonical cell CSV.

Drops any existing rows for that method in the canonical file first (so
re-runs are idempotent), then appends the method's rows from the temp
file. Both files must share the canonical schema (same harness).

Usage: python merge_method.py <canonical.csv> <temp.csv> <method>
"""
from __future__ import annotations

import csv
import sys


def main() -> int:
    canon, tmp, method = sys.argv[1], sys.argv[2], sys.argv[3]
    r = csv.DictReader(open(canon))
    fn = r.fieldnames
    if fn is None:
        print(f"  merge: {canon} has no header", file=sys.stderr)
        return 1
    kept = [row for row in r if row.get("method") != method]
    try:
        t = csv.DictReader(open(tmp))
        new = [row for row in t if row.get("method") == method]
    except FileNotFoundError:
        new = []
    if not new:
        print(f"  merge: no '{method}' rows in {tmp} (left as-is)")
        return 0
    w = csv.DictWriter(open(canon, "w"), fieldnames=fn, extrasaction="ignore")
    w.writeheader()
    w.writerows(kept + new)
    print(f"  merge: +{len(new)} '{method}' rows into {canon}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Decide whether a picker_demo cell needs all methods or only bs_kernel,
and under which KV dtype.

Reads a combined picker_demo CSV and prints one of:
    bsk_only bf16   — full baselines exist (paged + fasttree + deft).
                      Refresh bsk methods under bf16 KV.
    bsk_only fp8    — only paged baseline exists (fasttree + deft absent).
                      Likely a fp8-KV cell; refresh bsk methods under
                      fp8 KV.
    all bf16        — CSV missing/empty; run every method, starting
                      under bf16 KV.

Usage:
    python decide_methods.py <csv_path>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

BSK = {
    "bs_kernel",
    "bsk_2l_1p", "bs_kernel_2l1p",
    "bsk_2l_dt", "bs_kernel_2ldt",
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print("all bf16")
        return
    methods: set[str] = set()
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row.get("method", "")
            if m:
                methods.add(m)
    baselines = methods - BSK
    if not baselines:
        print("all bf16")
        return
    if "fasttree" in baselines or "deft" in baselines:
        print("bsk_only bf16")
    else:
        # Only paged (no fasttree/deft) → fp8-KV cell from a prior retry.
        print("bsk_only fp8")


if __name__ == "__main__":
    main()

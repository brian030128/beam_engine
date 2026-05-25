"""Verify, at a long decode, (A) bs_kernel is still the fastest *baseline*
and (B) the picker selects the right strategy per step (flips 1POOL→DEC_TAIL
near the measured ~4k crossover).

Usage:
    uv run python scripts/paper-exp/analyze_pick_correctness.py \
        <aggregate_csv> <step_timings_dir>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

BASELINES = {"paged", "fasttree", "deft", "mlca"}


def _best_decode(agg: Path) -> dict[str, float]:
    best: dict[str, float] = {}
    with open(agg) as f:
        for row in csv.DictReader(f):
            m = row["method"]
            dt = float(row["decode_total_ms"])
            if m not in best or dt < best[m]:
                best[m] = dt
    return best


def _load_strategy(path: Path) -> list[tuple[int, str]]:
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append((int(row["step"]), row.get("strategy", "")))
    return out


def main() -> None:
    agg = Path(sys.argv[1])
    sdir = Path(sys.argv[2])
    best = _best_decode(agg)

    # ---- A: fastest baseline? ----
    print("=== A: bs_kernel vs baselines (best decode_total ms) ===")
    ranked = sorted(best.items(), key=lambda kv: kv[1])
    for m, dt in ranked:
        tag = " <- PICKER" if m == "bs_kernel" else (" (baseline)" if m in BASELINES else "")
        print(f"  {m:<20} {dt:12.1f}{tag}")
    if "bs_kernel" in best:
        bk = best["bs_kernel"]
        present_bl = {m: best[m] for m in BASELINES if m in best}
        if present_bl:
            fastest_bl = min(present_bl, key=present_bl.get)
            print(f"\n  bs_kernel={bk:.1f}  fastest baseline={fastest_bl}={present_bl[fastest_bl]:.1f}"
                  f"  -> bs_kernel {'FASTER' if bk < present_bl[fastest_bl] else 'SLOWER'} "
                  f"by {abs(present_bl[fastest_bl]-bk)/bk*100:.1f}%")
        # vs its own forced/scheduled references
        for ref in ("bsk_2l_1p_single", "bsk_2l_dt", "bsk_sched_1p_dt"):
            if ref in best:
                print(f"  vs {ref:<18} {best[ref]:12.1f}  (picker/ref = {bk/best[ref]:.4f})")

    # ---- B: picker strategy per step ----
    print("\n=== B: picker strategy schedule (bs_kernel) ===")
    bkpath = sdir / "self_consistency__bs_kernel.csv"
    if not bkpath.exists():
        print(f"  (no per-step dump at {bkpath})")
        return
    seq = _load_strategy(bkpath)
    from collections import Counter
    n = len(seq)
    counts = Counter(s for _, s in seq)
    print(f"  picker decision — strategy share over {n} steps:")
    for s, c in counts.most_common():
        print(f"    {s:<26} {c:>6} steps   {c / n * 100:5.1f}%")
    # Detect transitions.
    transitions = []
    prev = None
    for step, s in seq:
        if s != prev:
            transitions.append((step, s))
            prev = s
    print("  transitions (step -> strategy):")
    for step, s in transitions:
        print(f"    @{step:>6}  {s}")
    if len(transitions) == 1:
        print("  -> picker NEVER switches strategy (single strategy whole run).")
        only = transitions[0][1]
        if "1pool" in only:
            print("     It stayed on 1POOL — MIS-PICK for long tails (>~4k); the")
            print("     bsk_sched/forced-dt refs above should beat it if so.")
        elif "dec_tail" in only:
            print("     It stayed on DEC_TAIL — leaves the short-tail 1POOL win on")
            print("     the table for steps <~4k.")
    else:
        print(f"  -> picker switches {len(transitions)-1} time(s); first non-initial "
              f"switch @step {transitions[1][0]} (measured forward crossover ~4000).")


if __name__ == "__main__":
    main()

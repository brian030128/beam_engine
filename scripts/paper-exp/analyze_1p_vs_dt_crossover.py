"""Analyse the 1POOL-vs-DEC_TAIL forward-time-vs-tail-length crossover.

Reads the two per-step timing dumps written by bench_sglang_e2e.py
(BE_DUMP_STEP_TIMINGS) for the forced bsk_2l_1p_single / bsk_2l_dt runs,
buckets the per-step forward (and total decode-step) times by decode position,
fits a line to each (forward vs step), reports per-bucket means, the measured
crossover step (first bucket where dt forward < 1pool forward), and the
slope-implied crossover. Usage:

    uv run python scripts/paper-exp/analyze_1p_vs_dt_crossover.py <step_timings_dir>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def _load(path: Path) -> tuple[list[int], list[float], list[float]]:
    steps, fwd, tot = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            fwd.append(float(row["forward_ms"]) if row["forward_ms"] else 0.0)
            tot.append(float(row["decode_step_ms"]) if row["decode_step_ms"] else 0.0)
    return steps, fwd, tot


def _lin_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope, intercept for y = slope*x + intercept."""
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _bucket_means(steps, vals, width: int) -> list[tuple[int, float]]:
    """Mean of vals per [k*width, (k+1)*width) bucket; returns (bucket_mid, mean)."""
    buckets: dict[int, list[float]] = {}
    for s, v in zip(steps, vals):
        buckets.setdefault(s // width, []).append(v)
    out = []
    for k in sorted(buckets):
        b = buckets[k]
        out.append((k * width + width // 2, sum(b) / len(b)))
    return out


def main() -> None:
    d = Path(sys.argv[1])
    p1 = d / "self_consistency__bsk_2l_1p_single.csv"
    pd = d / "self_consistency__bsk_2l_dt.csv"
    if not p1.exists() or not pd.exists():
        sys.exit(f"missing dumps: {p1.exists()=} {pd.exists()=} in {d}")

    s1, f1, t1 = _load(p1)
    sd, fd, td = _load(pd)
    n = min(len(s1), len(sd))
    print(f"steps: 1pool={len(s1)} dt={len(sd)} (using first {n})")

    W = 1000
    b1 = _bucket_means(s1[:n], f1[:n], W)
    bd = _bucket_means(sd[:n], fd[:n], W)
    bd_map = dict(bd)
    t1b = dict(_bucket_means(s1[:n], t1[:n], W))
    tdb = dict(_bucket_means(sd[:n], td[:n], W))

    print(f"\nper-{W}-step bucket means (forward ms/step):")
    print(f"{'step~':>8} {'1pool_fwd':>10} {'dt_fwd':>10} {'dt-1p':>8} "
          f"{'1pool_tot':>10} {'dt_tot':>10}")
    crossover = None
    for mid, v1 in b1:
        vd = bd_map.get(mid, float("nan"))
        diff = vd - v1
        if crossover is None and vd < v1:
            crossover = mid
        print(f"{mid:>8} {v1:>10.3f} {vd:>10.3f} {diff:>8.3f} "
              f"{t1b.get(mid, float('nan')):>10.3f} {tdb.get(mid, float('nan')):>10.3f}")

    # Linear fits on the raw per-step forward (skip first 50 warmup-ish steps).
    sk = 50
    m1, c1 = _lin_fit([float(x) for x in s1[sk:n]], f1[sk:n])
    md, cd = _lin_fit([float(x) for x in sd[sk:n]], fd[sk:n])
    print(f"\nlinear fit forward_ms = slope*step + intercept (steps {sk}..{n}):")
    print(f"  1pool: slope={m1*1000:.4f} ms/1k-steps  intercept={c1:.3f} ms")
    print(f"  dt   : slope={md*1000:.4f} ms/1k-steps  intercept={cd:.3f} ms")
    if md < m1:
        x_star = (c1 - cd) / (md - m1)
        print(f"  -> dt slope shallower; slope-implied crossover at step "
              f"{x_star:.0f}" + ("  (BEYOND measured range)" if x_star > n else "  (within range)"))
    else:
        print(f"  -> dt slope NOT shallower (Δslope={(md-m1)*1000:.4f} ms/1k); "
              f"lines diverge/parallel -> dt never overtakes 1pool on forward.")

    print(f"\nmeasured forward crossover bucket: "
          f"{crossover if crossover is not None else 'NONE (dt slower at every bucket)'}")

    print(f"\ncumulative decode_total (ms) at step checkpoints:")
    for cp in (1000, 2000, 4000, 8000, 12000, 16000):
        if cp <= n:
            print(f"  step {cp:>6}: 1pool={sum(t1[:cp]):10.1f}  dt={sum(td[:cp]):10.1f}  "
                  f"dt/1pool={sum(td[:cp])/max(sum(t1[:cp]),1e-9):.4f}")


if __name__ == "__main__":
    main()

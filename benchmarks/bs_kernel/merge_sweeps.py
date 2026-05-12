"""Merge two sweep_paper output directories.

Use case: rerun only the methods that changed (e.g. bs_kernel after a
strategy-set tweak) and merge with the prior baseline run. The merger:
  * Loads ``sweep_main.csv`` from both dirs.
  * Keeps every row from ``--new`` (overrides on duplicate
    (method, K, L_p, B, max_new)).
  * Fills with non-conflicting rows from ``--baseline``.
  * Concatenates ``sweep_picks.csv`` similarly.
  * Re-emits ``sweep_summary.md`` and ``sweep_winners.md`` from the
    merged CSV by re-running the same logic embedded here.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


KEY_FIELDS = ("method", "K", "L_p", "B", "max_new")


def _load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _row_key(r: dict) -> tuple:
    return tuple(r[k] for k in KEY_FIELDS)


def _merge_main(baseline_rows, new_rows):
    new_keys = {_row_key(r) for r in new_rows}
    merged = list(new_rows)
    for r in baseline_rows:
        if _row_key(r) not in new_keys:
            merged.append(r)
    # Sort by (L_p, K, B, method).
    merged.sort(key=lambda r: (
        int(r["L_p"]), int(r["K"]), int(r["B"]), r["method"],
    ))
    return merged


def _merge_picks(baseline_rows, new_rows):
    # Picks are keyed by (method, K, L_p, B, max_new, strategy).
    def k(r):
        return (r["method"], r["K"], r["L_p"], r["B"], r["max_new"], r["strategy"])
    # Drop any baseline picks for methods present in new (we replan from scratch).
    new_methods = {r["method"] for r in new_rows}
    merged = list(new_rows)
    for r in baseline_rows:
        if r["method"] not in new_methods:
            merged.append(r)
    merged.sort(key=lambda r: (
        int(r["L_p"]), int(r["K"]), int(r["B"]), r["method"], r["strategy"],
    ))
    return merged


def _emit_summary_md(main_rows, picks_rows, out_md: Path):
    by_lp_mode: dict = defaultdict(list)
    for r in main_rows:
        if r["status"] != "OK":
            continue
        mode = "dbs" if r["method"].startswith("dbs_") else "std"
        by_lp_mode[(int(r["L_p"]), mode)].append(r)

    by_cell: dict = defaultdict(list)
    for p in picks_rows:
        method = p["method"]
        K = int(p["K"]); L_p = int(p["L_p"]); B = int(p["B"])
        n_steps = int(p["n_steps"]); count = int(p["count"])
        strat, depth, pool, tlarge = (
            p["strategy"], p["depth"], p["pool"], p["t_large"],
        )
        by_cell[(method, K, L_p, B)].append(
            (strat, depth, pool, tlarge, count, n_steps),
        )
    picks_dominant: dict = {}
    for key, entries in by_cell.items():
        entries.sort(key=lambda e: -e[4])
        picks_dominant[key] = entries[0]

    lines: list[str] = []
    lines.append("# Cross-method sweep — paper results (merged)\n")
    lines.append(
        "Model: `meta-llama/Llama-3.2-1B` (fp16). H100 80GB. "
        "Same prompt repeated B times.\n"
    )
    lines.append("OOM cells omitted.\n")

    for (L_p, mode), rows in sorted(by_lp_mode.items()):
        method_set = sorted({r["method"] for r in rows})
        Ks = sorted({int(r["K"]) for r in rows})
        Bs = sorted({int(r["B"]) for r in rows})
        lines.append(f"\n## L_p={L_p}, mode={mode} — ms/step\n")
        header_cells = ["K\\B"] + [f"B={B}" for B in Bs]
        for method in method_set:
            lines.append(f"\n### {method}\n")
            lines.append("| " + " | ".join(header_cells) + " |")
            lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
            for K in Ks:
                row = [f"K={K}"]
                for B in Bs:
                    cell = next(
                        (r for r in rows
                         if r["method"] == method and int(r["K"]) == K
                         and int(r["B"]) == B),
                        None,
                    )
                    row.append(
                        f"{float(cell['decode_per_token_ms']):.2f}"
                        if cell else "—"
                    )
                lines.append("| " + " | ".join(row) + " |")

    lines.append("\n\n## bs_kernel dominant pick per cell\n")
    lines.append("Format: ``strategy (depth, pool, t_large) — N/255 steps``\n")
    for L_p in sorted({int(r["L_p"]) for r in main_rows}):
        for bs_method in ("bs_kernel", "dbs_bs_kernel"):
            lines.append(f"\n### L_p={L_p}, {bs_method}\n")
            Ks = sorted({k for (m, k, lp, b) in picks_dominant
                         if m == bs_method and lp == L_p})
            Bs = sorted({b for (m, k, lp, b) in picks_dominant
                         if m == bs_method and lp == L_p})
            if not Ks:
                lines.append("(no successful cells)")
                continue
            header_cells = ["K\\B"] + [f"B={B}" for B in Bs]
            lines.append("| " + " | ".join(header_cells) + " |")
            lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
            for K in Ks:
                row = [f"K={K}"]
                for B in Bs:
                    entry = picks_dominant.get((bs_method, K, L_p, B))
                    if entry is None:
                        row.append("—")
                    else:
                        strat, depth, pool, tlarge, count, n_steps = entry
                        if strat == "PER_BEAM":
                            label = f"PER_BEAM {count}/{n_steps}"
                        else:
                            label = f"{strat} ({depth},{pool},{tlarge}) {count}/{n_steps}"
                        row.append(label)
                lines.append("| " + " | ".join(row) + " |")

    out_md.write_text("\n".join(lines) + "\n")


def _emit_winners_md(main_rows, out_md: Path):
    all_cells: dict = defaultdict(dict)
    for r in main_rows:
        if r["status"] != "OK":
            continue
        mode = "dbs" if r["method"].startswith("dbs_") else "std"
        m = r["method"].replace("dbs_", "")
        key = (int(r["L_p"]), mode, int(r["K"]), int(r["B"]))
        all_cells[key][m] = float(r["decode_per_token_ms"])

    lines = ["# Fastest method per cell (with speedup over runner-up)\n"]
    lines.append("Model: meta-llama/Llama-3.2-1B (fp16). H100 80GB. ms/step.\n")
    for L_p in sorted({lp for (lp, _m, _k, _b) in all_cells}):
        for mode in ("std", "dbs"):
            lines.append(f"\n## L_p={L_p}, mode={mode}\n")
            Ks = sorted({k for (lp, mo, k, b) in all_cells
                         if lp == L_p and mo == mode})
            Bs = sorted({b for (lp, mo, k, b) in all_cells
                         if lp == L_p and mo == mode})
            lines.append("| K\\B | " + " | ".join(f"B={b}" for b in Bs) + " |")
            lines.append("|" + "|".join(["---"] * (len(Bs) + 1)) + "|")
            for K in Ks:
                row = [f"K={K}"]
                for B in Bs:
                    cell = all_cells.get((L_p, mode, K, B), {})
                    if not cell:
                        row.append("—")
                        continue
                    runners = sorted(cell.items(), key=lambda x: x[1])
                    winner, ms = runners[0]
                    if len(runners) >= 2:
                        speedup = runners[1][1] / runners[0][1]
                        row.append(f"**{winner}** {ms:.2f} ({speedup:.2f}×)")
                    else:
                        row.append(f"**{winner}** {ms:.2f}")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
    out_md.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path,
                    help="Prior sweep output dir (sweep_main.csv + sweep_picks.csv)")
    ap.add_argument("--new", required=True, type=Path,
                    help="New sweep output dir (subset of methods)")
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    bm = _load_csv(args.baseline / "sweep_main.csv")
    nm = _load_csv(args.new / "sweep_main.csv")
    bp = _load_csv(args.baseline / "sweep_picks.csv")
    np = _load_csv(args.new / "sweep_picks.csv")

    merged_main = _merge_main(bm, nm)
    merged_picks = _merge_picks(bp, np)

    main_csv = args.out_dir / "sweep_main.csv"
    picks_csv = args.out_dir / "sweep_picks.csv"
    with main_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bm[0].keys())
        w.writeheader()
        w.writerows(merged_main)
    with picks_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bp[0].keys())
        w.writeheader()
        w.writerows(merged_picks)

    _emit_summary_md(merged_main, merged_picks, args.out_dir / "sweep_summary.md")
    _emit_winners_md(merged_main, args.out_dir / "sweep_winners.md")

    print(f"merged {len(merged_main)} main rows, {len(merged_picks)} picks rows")
    print(f"wrote {main_csv}")
    print(f"wrote {picks_csv}")
    print(f"wrote {args.out_dir / 'sweep_summary.md'}")
    print(f"wrote {args.out_dir / 'sweep_winners.md'}")


if __name__ == "__main__":
    main()

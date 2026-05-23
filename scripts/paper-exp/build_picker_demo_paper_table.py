"""Build the final paper table for picker_demo across all three models.

Reads every picker_demo cell CSV and emits one consolidated table:
  - Markdown to stdout (and optionally a file via --out-md).
  - CSV via --out-csv.

Columns per row:
  model, ID, scenario, K, B, max_new,
  decode_p50_ms for {paged, fasttree, deft, bs_kernel, 1POOL (forced), DT (forced)},
  speedup ratios: bs/paged, bs/fasttree, bs/deft,
  picker regime: 1p/dt ratio, picker_pick, picker_oracle_ratio, picker_match.

Run *after* the three sbatch jobs finish:
    uv run python scripts/paper-exp/build_picker_demo_paper_table.py \
        --out-md docs/exp_picker_demo_paper_table.md \
        --out-csv benchmarks/bs_kernel/results/paper-exp/picker_demo/paper_table.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "benchmarks/bs_kernel/results/paper-exp/picker_demo"

PICKER_ALIASES = {"bs_kernel"}
PAGED_ALIASES = {"paged"}
FASTTREE_ALIASES = {"fasttree"}
DEFT_ALIASES = {"deft"}
FUSED_ALIASES = {"bsk_2l_1p", "bs_kernel_2l1p"}
DECTAIL_ALIASES = {"bsk_2l_dt", "bs_kernel_2ldt"}

# (tag, scenario, K, B, mn, Lp) per model. Lp=None means sglang CSV;
# Lp set means bench_batched CSV.
MODEL_CELLS: dict[str, list[tuple]] = {
    "Llama-3.2-1B-FP8": [
        ("A1", "beam_search",           4,  1,  64,  40000),
        ("A2", "multi_chain_reasoning", 4,  1,  32,  None),
        ("A3", "multi_doc_qa",          32, 1,  256, None),
        ("A4", "multi_few_shot",        16, 8,  256, None),
        ("A5", "multi_level_system",    32, 4,  256, None),
        ("D1", "multi_chain_reasoning", 32, 32, 256, None),
        ("D2", "multi_few_shot",        16, 32, 256, None),
        ("D3", "multi_level_system",    32, 16, 256, None),
        ("D4", "multi_doc_qa",          16, 4,  256, None),
    ],
    "Llama-3.1-8B-FP8": [
        ("B1", "beam_search",           4,  1,  128, 40000),
        ("B2", "multi_chain_reasoning", 4,  1,  32,  None),
        ("B3", "multi_doc_qa",          64, 1,  256, None),
        ("B4", "multi_few_shot",        32, 4,  256, None),
        ("B5", "multi_level_system",    64, 4,  256, None),
        ("B6", "multi_chain_reasoning", 64, 32, 256, None),
        ("D1", "multi_chain_reasoning", 32, 32, 256, None),
        ("D2", "multi_few_shot",        16, 32, 256, None),
        ("D3", "multi_level_system",    32, 16, 256, None),
        ("D4", "multi_doc_qa",          16, 4,  256, None),
    ],
    "Llama-3-70B-Instruct-FP8": [
        ("C1", "beam_search",           4,  1,  64,  40000),
        ("C2", "multi_chain_reasoning", 4,  1,  32,  None),
        ("C3", "multi_doc_qa",          32, 1,  256, None),
        ("C4", "multi_few_shot",        16, 4,  256, None),
        ("C5", "multi_level_system",    32, 4,  256, None),
        ("C6", "multi_chain_reasoning", 64, 4,  256, None),
        ("D1", "multi_chain_reasoning", 32, 32, 256, None),
        ("D2", "multi_few_shot",        16, 32, 256, None),
        ("D3", "multi_level_system",    32, 16, 256, None),
        ("D4", "multi_doc_qa",          16, 4,  256, None),
    ],
}


def _csv_path(model_dir: Path, cell: tuple) -> Path:
    tag, scenario, K, B, mn, Lp = cell
    if Lp is None:
        return model_dir / f"{tag}_{scenario}_K{K}_B{B}_mn{mn}.csv"
    return model_dir / f"{tag}_beam_search_K{K}_B{B}_Lp{Lp}_mn{mn}.csv"


def _decode_p50_ms(csv_path: Path) -> dict[str, float]:
    """{method: median across repeats of decode_p50_ms} for a cell."""
    if not csv_path.exists():
        return {}
    by_method: dict[str, list[float]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row.get("method", "")
            v_str = row.get("decode_p50_ms") or row.get("decode_total_ms")
            try:
                v = float(v_str)
            except (TypeError, ValueError):
                continue
            if not row.get("decode_p50_ms"):
                # bench_batched.py only has decode_total_ms — derive p50.
                try:
                    n_steps = float(row.get("n_decode_steps") or "0")
                    if n_steps > 0:
                        v = v / n_steps
                except ValueError:
                    continue
            by_method.setdefault(m, []).append(v)
    return {m: sorted(xs)[len(xs) // 2] for m, xs in by_method.items()}


def _pick(totals: dict[str, float], aliases: set[str]) -> float | None:
    for a in aliases:
        if a in totals:
            return totals[a]
    return None


def collate_cell(model_tag: str, model_dir: Path, cell: tuple) -> dict:
    tag = cell[0]
    csv_path = _csv_path(model_dir, cell)
    totals = _decode_p50_ms(csv_path)
    paged = _pick(totals, PAGED_ALIASES)
    fasttree = _pick(totals, FASTTREE_ALIASES)
    deft = _pick(totals, DEFT_ALIASES)
    picker = _pick(totals, PICKER_ALIASES)
    fused = _pick(totals, FUSED_ALIASES)
    dectail = _pick(totals, DECTAIL_ALIASES)
    r = {
        "model": model_tag, "id": tag, "scenario": cell[1],
        "K": cell[2], "B": cell[3], "mn": cell[4],
        "csv_exists": csv_path.exists(),
        "paged": paged, "fasttree": fasttree, "deft": deft,
        "picker": picker, "fused_1p": fused, "dectail_dt": dectail,
    }
    if fused is not None and dectail is not None:
        r["1p_over_dt"] = fused / dectail
        oracle = min(fused, dectail)
        r["oracle"] = oracle
        if picker is not None:
            r["picker_match"] = picker <= oracle * 1.01
            r["picker_over_oracle"] = picker / oracle
            d_1p = abs(picker - fused)
            d_dt = abs(picker - dectail)
            r["picker_pick"] = "1POOL" if d_1p < d_dt else "DT"
    if picker is not None and paged is not None:
        r["bs_over_paged"] = picker / paged
    if picker is not None and fasttree is not None:
        r["bs_over_fasttree"] = picker / fasttree
    if picker is not None and deft is not None:
        r["bs_over_deft"] = picker / deft
    return r


def _fmt(v, fmt: str = "{:.2f}") -> str:
    if v is None:
        return "—"
    return fmt.format(v)


def render_md(all_rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Picker-demo paper table (combined across all models)\n")
    lines.append(
        "Each row is one cell. Times are median-across-repeats of "
        "`decode_p50_ms` (per-decode-step latency). Speedup columns are "
        "`bs_kernel / baseline` — values < 1.0 mean bs_kernel wins.\n"
    )
    lines.append(
        "| Model | ID | Scenario | K | B | mn | "
        "paged | fasttree | deft | bs_kernel | 1POOL | DT | "
        "bs/paged | bs/ft | bs/deft | 1p/dt | picker | match |"
    )
    lines.append(
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|---|---|"
    )
    cur_model: str | None = None
    for r in all_rows:
        if r["model"] != cur_model:
            cur_model = r["model"]
        if not r["csv_exists"]:
            lines.append(
                f"| {r['model']} | **{r['id']}** | {r['scenario']} | "
                f"{r['K']} | {r['B']} | {r['mn']} | — | — | — | — | — | — | "
                f"— | — | — | — | — | _missing_ |"
            )
            continue
        match = r.get("picker_match")
        match_cell = "✓" if match else ("✗" if match is False else "—")
        lines.append(
            f"| {r['model']} | **{r['id']}** | {r['scenario']} | "
            f"{r['K']} | {r['B']} | {r['mn']} | "
            f"{_fmt(r.get('paged'))} | {_fmt(r.get('fasttree'))} | "
            f"{_fmt(r.get('deft'))} | {_fmt(r.get('picker'))} | "
            f"{_fmt(r.get('fused_1p'))} | {_fmt(r.get('dectail_dt'))} | "
            f"{_fmt(r.get('bs_over_paged'))} | "
            f"{_fmt(r.get('bs_over_fasttree'))} | "
            f"{_fmt(r.get('bs_over_deft'))} | "
            f"{_fmt(r.get('1p_over_dt'), '{:.3f}')} | "
            f"{r.get('picker_pick', '—')} | {match_cell} |"
        )

    wins_paged = [r["bs_over_paged"] for r in all_rows
                  if r.get("bs_over_paged") is not None]
    wins_ft = [r["bs_over_fasttree"] for r in all_rows
               if r.get("bs_over_fasttree") is not None]
    wins_deft = [r["bs_over_deft"] for r in all_rows
                 if r.get("bs_over_deft") is not None]
    match_rows = [r for r in all_rows if "picker_match" in r]
    matched = [r for r in match_rows if r["picker_match"]]

    def _stats(xs: list[float]) -> str:
        if not xs:
            return "—"
        n = len(xs)
        better = sum(1 for x in xs if x < 1.0)
        return f"{n} cells, {better} bs_kernel wins, geomean={_geomean(xs):.3f}"

    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- **bs/paged**: {_stats(wins_paged)}")
    lines.append(f"- **bs/fasttree**: {_stats(wins_ft)}")
    lines.append(f"- **bs/deft**: {_stats(wins_deft)}")
    if match_rows:
        lines.append(
            f"- **picker match**: {len(matched)}/{len(match_rows)} cells "
            f"within 1% of oracle"
        )
        worst = max(match_rows, key=lambda r: r.get("picker_over_oracle", 1.0))
        lines.append(
            f"  - worst miss: {worst['model']} / {worst['id']} "
            f"({worst['scenario']}) at "
            f"{worst.get('picker_over_oracle', float('nan')):.3f}× oracle"
        )
    return "\n".join(lines) + "\n"


def _geomean(xs: list[float]) -> float:
    from math import exp, log
    return exp(sum(log(x) for x in xs) / len(xs))


def render_csv(all_rows: list[dict]) -> str:
    cols = [
        "model", "id", "scenario", "K", "B", "mn",
        "paged_ms", "fasttree_ms", "deft_ms", "bs_kernel_ms",
        "fused_1p_ms", "dectail_dt_ms",
        "bs_over_paged", "bs_over_fasttree", "bs_over_deft",
        "fused_over_dectail", "picker_pick", "picker_over_oracle",
        "picker_match",
    ]
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(cols)
    for r in all_rows:
        writer.writerow([
            r["model"], r["id"], r["scenario"], r["K"], r["B"], r["mn"],
            r.get("paged"), r.get("fasttree"), r.get("deft"),
            r.get("picker"), r.get("fused_1p"), r.get("dectail_dt"),
            r.get("bs_over_paged"), r.get("bs_over_fasttree"),
            r.get("bs_over_deft"), r.get("1p_over_dt"),
            r.get("picker_pick"), r.get("picker_over_oracle"),
            r.get("picker_match"),
        ])
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for model_tag, cells in MODEL_CELLS.items():
        model_dir = RESULTS_ROOT / model_tag
        if not model_dir.exists():
            for c in cells:
                all_rows.append({
                    "model": model_tag, "id": c[0], "scenario": c[1],
                    "K": c[2], "B": c[3], "mn": c[4], "csv_exists": False,
                })
            continue
        for c in cells:
            all_rows.append(collate_cell(model_tag, model_dir, c))

    md = render_md(all_rows)
    print(md)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md)
        print(f"# wrote markdown to {args.out_md}", file=__import__("sys").stderr)
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        args.out_csv.write_text(render_csv(all_rows))
        print(f"# wrote csv to {args.out_csv}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()

"""Collate picker_demo (5-scenario × 3-model) CSVs into a Measured
results table, written back into ``docs/exp_picker_demo.md`` between
the ``<!-- BEGIN MEASURED -->`` / ``<!-- END MEASURED -->`` markers.

Per cell the script computes:

  decode_p50_ms(method) for {paged, bs_kernel, bsk_2l_1p, bsk_2l_dt}
  1p/dt   = decode_p50(bsk_2l_1p) / decode_p50(bsk_2l_dt)
  bs/paged = decode_p50(bs_kernel) / decode_p50(paged)
  oracle  = min(decode_p50(bsk_2l_1p), decode_p50(bsk_2l_dt))
  picker_match = bs_kernel <= oracle * 1.01
  picker_pick  = "1POOL" if bs_kernel closer to 2l1p else "DT"

CSV filenames follow the sbatch convention:
  {tag}_{scenario}_K{K}_B{B}_mn{mn}.csv         (sglang cells)
  {tag}_beam_search_K{K}_B{B}_Lp{Lp}_mn{mn}.csv (bench_batched cell)

Usage:
    uv run python scripts/paper-exp/collate_picker_demo.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "benchmarks/bs_kernel/results/paper-exp/picker_demo"
DOC_PATH = REPO_ROOT / "docs/exp_picker_demo.md"

PICKER_ALIASES = {"bs_kernel"}
PAGED_ALIASES = {"paged"}
FASTTREE_ALIASES = {"fasttree"}
DEFT_ALIASES = {"deft"}
FUSED_ALIASES = {"bsk_2l_1p", "bs_kernel_2l1p"}
DECTAIL_ALIASES = {"bsk_2l_dt", "bs_kernel_2ldt"}

# Per-model cell registry: (tag, scenario, K, B, mn) — keep in sync
# with the sbatch templates.
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

# Predicted regime per cell (for the "match prediction?" column).
# D-cells share the same (tag → pred) mapping across models, since
# the regime is dominated by R = B*K and L_priv, not model size.
PRED: dict[str, tuple[str, float]] = {
    "A1": ("1POOL", 0.94), "A2": ("1POOL", 0.94), "A3": ("DT", 1.18),
    "A4": ("DT", 1.22), "A5": ("DT", 1.18),
    "B1": ("1POOL", 0.94), "B2": ("1POOL", 0.94), "B3": ("DT", 1.30),
    "B4": ("DT", 1.25), "B5": ("DT", 1.28), "B6": ("DT", 1.35),
    "C1": ("1POOL", 0.96), "C2": ("1POOL", 0.96), "C3": ("DT", 1.12),
    "C4": ("DT", 1.12), "C5": ("DT", 1.12), "C6": ("DT", 1.15),
    "D1": ("DT", 1.20), "D2": ("DT", 1.22),
    "D3": ("DT", 1.20), "D4": ("DT", 1.15),
}


def _csv_path(model_dir: Path, cell: tuple) -> Path:
    tag, scenario, K, B, mn, Lp = cell
    if Lp is None:
        return model_dir / f"{tag}_{scenario}_K{K}_B{B}_mn{mn}.csv"
    return model_dir / f"{tag}_beam_search_K{K}_B{B}_Lp{Lp}_mn{mn}.csv"


def _decode_p50_ms(csv_path: Path) -> dict[str, float]:
    """Return {method: median decode_p50_ms across repeats} for a cell."""
    if not csv_path.exists():
        return {}
    by_method: dict[str, list[float]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row.get("method", "")
            # bench_batched.py uses "decode_p50_ms" if available, else
            # decode_total_ms / n_decode_steps.
            v_str = row.get("decode_p50_ms") or row.get("decode_total_ms")
            try:
                v = float(v_str)
            except (TypeError, ValueError):
                continue
            if not row.get("decode_p50_ms"):
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


def collate_cell(model_dir: Path, cell: tuple) -> dict:
    tag = cell[0]
    csv_path = _csv_path(model_dir, cell)
    totals = _decode_p50_ms(csv_path)
    paged = _pick(totals, PAGED_ALIASES)
    fasttree = _pick(totals, FASTTREE_ALIASES)
    deft = _pick(totals, DEFT_ALIASES)
    picker = _pick(totals, PICKER_ALIASES)
    fused = _pick(totals, FUSED_ALIASES)
    dectail = _pick(totals, DECTAIL_ALIASES)
    result = {
        "tag": tag, "scenario": cell[1], "K": cell[2], "B": cell[3],
        "mn": cell[4], "csv_exists": csv_path.exists(),
        "paged": paged, "fasttree": fasttree, "deft": deft,
        "picker": picker, "fused_1p": fused, "dectail_dt": dectail,
    }
    if fused is not None and dectail is not None:
        result["1p_over_dt"] = fused / dectail
        oracle = min(fused, dectail)
        result["oracle"] = oracle
        if picker is not None:
            result["picker_match"] = picker <= oracle * 1.01
            # pick which forced strategy the picker's decode_p50 is closer to
            d_1p = abs(picker - fused)
            d_dt = abs(picker - dectail)
            result["picker_pick"] = "1POOL" if d_1p < d_dt else "DT"
    if picker is not None and paged is not None:
        result["bs_over_paged"] = picker / paged
    if picker is not None and fasttree is not None:
        result["bs_over_fasttree"] = picker / fasttree
    if picker is not None and deft is not None:
        result["bs_over_deft"] = picker / deft
    return result


def _fmt(v, fmt: str = "{:.2f}") -> str:
    if v is None:
        return "—"
    return fmt.format(v)


def render_table(model_tag: str, results: list[dict]) -> str:
    lines = [f"### {model_tag}\n"]
    lines.append(
        "| ID | scenario | K | B | mn | "
        "paged | fasttree | deft | picker | 1POOL | DT | "
        "1p/dt | bs/paged | bs/ft | bs/deft | "
        "picker pick | picker match | regime |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|---|---|---|"
    )
    for r in results:
        if not r["csv_exists"]:
            lines.append(
                f"| {r['tag']} | {r['scenario']} | {r['K']} | {r['B']} | "
                f"{r['mn']} | — | — | — | — | — | — | — | — | — | — | — | — | "
                f"_missing CSV_ |"
            )
            continue
        pred_regime = PRED.get(r["tag"], ("?", 0.0))[0]
        actual_regime = "?"
        if "1p_over_dt" in r:
            actual_regime = "1POOL" if r["1p_over_dt"] < 1.0 else "DT"
        regime_match = "✓" if actual_regime == pred_regime else "✗"
        lines.append(
            f"| **{r['tag']}** | {r['scenario']} | {r['K']} | {r['B']} | "
            f"{r['mn']} | "
            f"{_fmt(r.get('paged'))} | {_fmt(r.get('fasttree'))} | "
            f"{_fmt(r.get('deft'))} | {_fmt(r.get('picker'))} | "
            f"{_fmt(r.get('fused_1p'))} | {_fmt(r.get('dectail_dt'))} | "
            f"{_fmt(r.get('1p_over_dt'), '{:.3f}')} | "
            f"{_fmt(r.get('bs_over_paged'), '{:.2f}')} | "
            f"{_fmt(r.get('bs_over_fasttree'), '{:.2f}')} | "
            f"{_fmt(r.get('bs_over_deft'), '{:.2f}')} | "
            f"{r.get('picker_pick', '—')} | "
            f"{'✓' if r.get('picker_match') else '✗' if 'picker_match' in r else '—'} | "
            f"{regime_match} ({pred_regime}→{actual_regime}) |"
        )
    return "\n".join(lines) + "\n"


def render_discussion(all_results: dict[str, list[dict]]) -> str:
    notes: list[str] = []
    for model_tag, results in all_results.items():
        for r in results:
            if not r["csv_exists"]:
                continue
            issues: list[str] = []
            if "picker_match" in r and not r["picker_match"]:
                ratio = r["picker"] / r["oracle"]
                issues.append(
                    f"picker {ratio:.3f}× of oracle "
                    f"(picked {r.get('picker_pick')}, "
                    f"forced 1POOL={r['fused_1p']:.2f} ms, "
                    f"forced DT={r['dectail_dt']:.2f} ms)"
                )
            if "1p_over_dt" in r:
                pred_regime, pred_ratio = PRED.get(r["tag"], ("?", 0.0))
                actual = r["1p_over_dt"]
                actual_regime = "1POOL" if actual < 1.0 else "DT"
                if actual_regime != pred_regime:
                    issues.append(
                        f"regime flipped: predicted {pred_regime} "
                        f"({pred_ratio}), measured {actual:.3f} "
                        f"({actual_regime})"
                    )
            if issues:
                notes.append(
                    f"- **{r['tag']}** ({model_tag} / {r['scenario']}): "
                    + "; ".join(issues)
                )
    if not notes:
        return "_No picker disagreements or regime flips — all cells matched prediction._\n"
    return "\n".join(notes) + "\n"


def main() -> None:
    if not RESULTS_ROOT.exists():
        print(f"results dir not found: {RESULTS_ROOT}")
        return

    all_results: dict[str, list[dict]] = {}
    rendered_tables: list[str] = []
    for model_tag, cells in MODEL_CELLS.items():
        model_dir = RESULTS_ROOT / model_tag
        if not model_dir.exists():
            print(f"  (skipping {model_tag}: directory missing)")
            continue
        results = [collate_cell(model_dir, c) for c in cells]
        all_results[model_tag] = results
        rendered_tables.append(render_table(model_tag, results))

    measured_block = "\n".join(rendered_tables) if rendered_tables else \
        "_(No results found — submit the three sbatch jobs first.)_\n"
    discussion_block = render_discussion(all_results)

    if not DOC_PATH.exists():
        print(f"doc not found: {DOC_PATH}")
        print(measured_block)
        return

    text = DOC_PATH.read_text()
    text = re.sub(
        r"<!-- BEGIN MEASURED -->.*?<!-- END MEASURED -->",
        f"<!-- BEGIN MEASURED -->\n{measured_block}\n<!-- END MEASURED -->",
        text, flags=re.DOTALL,
    )
    text = re.sub(
        r"<!-- BEGIN DISCUSSION -->.*?<!-- END DISCUSSION -->",
        f"<!-- BEGIN DISCUSSION -->\n{discussion_block}\n<!-- END DISCUSSION -->",
        text, flags=re.DOTALL,
    )
    DOC_PATH.write_text(text)
    print(f"updated {DOC_PATH}")


if __name__ == "__main__":
    main()

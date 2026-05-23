"""Build final_paper_e2e_table.md — picker_demo per-cell comparison with
all baseline methods, using:
  * paged / fasttree / deft / bs_kernel (picker) / 2ldt — from the
    original `paper-exp/picker_demo/<MODEL_TAG>/` runs (kernel change
    doesn't affect these paths).
  * bs_kernel_2l1p_single (forced SHARED_2L_1POOL +
    FusedMultiLevelCascadeAttentionWrapper(single_launch=True)) — from
    `paper-exp/picker_demo_2l1p_single_vs_2ldt/<MODEL_TAG>/`.

X1 / batched cells use ``decode_per_token_ms``. The remaining sglang
cells use ``decode_total_ms`` median over the 3 repeats.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import median

OLD = Path("benchmarks/bs_kernel/results/paper-exp/picker_demo")
NEW = Path("benchmarks/bs_kernel/results/paper-exp/picker_demo_2l1p_single_vs_2ldt")

MODELS = [
    # Each cell row's label includes L_p — the shared prefix length used
    # for that scenario. For batched cells (X1) L_p is set explicitly via
    # the bench CLI. For sglang cells, L_p comes from the L_shared the
    # harness derived from the prompts file (logged in run.log).
    ("Llama-3.2-1B (bf16 weights + bf16 KV, TP=1)", "Llama-3.2-1B-bf16", [
        ("A1", "batched", "A1_beam_search_K4_B1_Lp40000_mn64.csv",     "beam_search  K=4 B=1 L_p=40000 mn=64"),
        ("A2", "sglang",  "A2_multi_chain_reasoning_K4_B1_mn32.csv",   "multi_chain_reasoning  K=4 B=1 L_p≈2581 mn=32"),
        ("A3", "sglang",  "A3_multi_doc_qa_K32_B1_mn256.csv",          "multi_doc_qa  K=32 B=1 L_p≈86627 mn=256"),
        ("A4", "sglang",  "A4_multi_few_shot_K16_B8_mn256.csv",        "multi_few_shot  K=16 B=8 L_p≈5861 mn=256"),
        ("A5", "sglang",  "A5_multi_level_system_K32_B4_mn256.csv",    "multi_level_system  K=32 B=4 L_p≈2515 mn=256"),
    ]),
    ("Llama-3.1-8B (bf16 weights + bf16 KV, TP=1)", "Llama-3.1-8B-bf16", [
        ("B1", "batched", "B1_beam_search_K4_B1_Lp40000_mn128.csv",    "beam_search  K=4 B=1 L_p=40000 mn=128"),
        ("B2", "sglang",  "B2_multi_chain_reasoning_K4_B1_mn32.csv",   "multi_chain_reasoning  K=4 B=1 L_p≈2581 mn=32"),
        ("B3", "sglang",  "B3_multi_doc_qa_K64_B1_mn256.csv",          "multi_doc_qa  K=64 B=1 L_p≈86627 mn=256"),
        ("B4", "sglang",  "B4_multi_few_shot_K32_B4_mn256.csv",        "multi_few_shot  K=32 B=4 L_p≈5864 mn=256"),
        ("B5", "sglang",  "B5_multi_level_system_K64_B4_mn256.csv",    "multi_level_system  K=64 B=4 L_p≈2515 mn=256"),
    ]),
    # 70B fp8 (TP=2) re-run of the 8B B3/B4/B5 cells, all 6 methods.
    # Results landed in a sibling dir tagged ``_8b_b3b4b5`` so they don't
    # collide with the C/D picker_demo cells above.
    ("Llama-3-70B-Instruct-FP8 — 8B B3/B4/B5 cells (bf16 act + fp8_e4m3 KV, TP=2)",
     "Llama-3-70B-Instruct-FP8_8b_b3b4b5", [
        ("B3",      "sglang", "B3_multi_doc_qa_K64_B1_mn256.csv",          "multi_doc_qa  K=64 B=1 L_p≈86627 mn=256"),
        ("B4",      "sglang", "B4_multi_few_shot_K32_B4_mn256.csv",        "multi_few_shot  K=32 B=4 L_p≈5864 mn=256"),
        ("B4(B=8)", "sglang", "B4_B8_multi_few_shot_K32_B8_mn256.csv",     "multi_few_shot  K=32 B=8 L_p≈5860 mn=256"),
        ("B4(B=16)","sglang", "B4_B16_multi_few_shot_K32_B16_mn256.csv",   "multi_few_shot  K=32 B=16 L_p≈5860 mn=256 (OOM)"),
        ("B4(B=32)","sglang", "B4_B32_multi_few_shot_K32_B32_mn256.csv",   "multi_few_shot  K=32 B=32 L_p≈5860 mn=256 (OOM)"),
        ("B5",      "sglang", "B5_multi_level_system_K64_B4_mn256.csv",    "multi_level_system  K=64 B=4 L_p≈2515 mn=256"),
    ]),
    ("Llama-3-70B-Instruct-FP8 (bf16 act + fp8_e4m3 KV, TP=2)", "Llama-3-70B-Instruct-FP8", [
        ("C1", "batched", "C1_beam_search_K4_B1_Lp40000_mn64.csv",     "beam_search  K=4 B=1 L_p=40000 mn=64"),
        ("C2", "sglang",  "C2_multi_chain_reasoning_K4_B1_mn32.csv",   "multi_chain_reasoning  K=4 B=1 L_p≈2581 mn=32"),
        ("C3", "sglang",  "C3_multi_doc_qa_K32_B1_mn256.csv",          "multi_doc_qa  K=32 B=1 L_p≈86627 mn=256"),
        ("C4", "sglang",  "C4_multi_few_shot_K16_B4_mn256.csv",        "multi_few_shot  K=16 B=4 L_p≈5859 mn=256"),
        ("C5", "sglang",  "C5_multi_level_system_K32_B4_mn256.csv",    "multi_level_system  K=32 B=4 L_p≈2515 mn=256"),
        ("C6", "sglang",  "C6_multi_chain_reasoning_K64_B4_mn256.csv", "multi_chain_reasoning  K=64 B=4 L_p≈2561 mn=256"),
        ("D1", "sglang",  "D1_multi_chain_reasoning_K32_B32_mn256.csv","multi_chain_reasoning  K=32 B=32 L_p≈2574 mn=256"),
        ("D2", "sglang",  "D2_multi_few_shot_K16_B32_mn256.csv",       "multi_few_shot  K=16 B=32 L_p≈5840 mn=256"),
        ("D3", "sglang",  "D3_multi_level_system_K32_B16_mn256.csv",   "multi_level_system  K=32 B=16 L_p≈2515 mn=256"),
        ("D4", "sglang",  "D4_multi_doc_qa_K16_B4_mn256.csv",          "multi_doc_qa  K=16 B=4 L_p≈88441 mn=256"),
    ]),
]

# Which method names appear in (a) batched bench output, (b) sglang output.
BATCHED_NAMES = {
    "paged": "paged", "fasttree": "fasttree", "deft": "deft",
    "bs_kernel": "bs_kernel",
    "2l1p_single": "bs_kernel_2l1p_single",
    "2ldt": "bs_kernel_2ldt",
}
SGLANG_NAMES = {
    "paged": "paged", "fasttree": "fasttree", "deft": "deft",
    "bs_kernel": "bs_kernel",
    "2l1p_single": "bsk_2l_1p_single",
    "2ldt": "bsk_2l_dt",
}


def load_batched(path: Path, method: str):
    if not path.exists():
        return None
    for r in csv.DictReader(open(path)):
        if r.get("method") == method:
            try:
                return float(r["decode_per_token_ms"])
            except (KeyError, ValueError):
                return None
    return None


def load_sglang(path: Path, method: str):
    if not path.exists():
        return None
    vals = []
    for r in csv.DictReader(open(path)):
        if r.get("method") != method:
            continue
        try:
            vals.append(float(r["decode_total_ms"]))
        except (KeyError, ValueError):
            pass
    return median(vals) if vals else None


def fmt(x):
    return f"{x:.2f}" if x is not None else "—"


def speedup(num, den):
    """Return num/den as 'X.XXx', or '—'."""
    if num is None or den is None or den == 0:
        return "—"
    return f"{num/den:.2f}×"


def best(values: dict) -> str | None:
    """Return the method name with the lowest time (excluding None)."""
    valid = [(k, v) for k, v in values.items() if v is not None]
    if not valid:
        return None
    return min(valid, key=lambda kv: kv[1])[0]


def render_model(header: str, model_tag: str, cells, only: list[str] | None = None):
    lines = [f"## {header}", ""]
    if only is not None and model_tag not in only:
        lines.append(f"*(deferred — awaiting run completion)*")
        lines.append("")
        return lines
    lines.append("Metric: cell `X1` is `decode_per_token_ms` (batched harness, hotpotqa). All other cells are sglang `decode_total_ms`, median over 3 repeats. Lower is better. **bold** marks the winner per row.")
    lines.append("")
    lines.append("| cell | scenario / config | paged | fasttree | deft | bs_kernel (picker) | 2ldt | **2l1p_single** | speedup vs paged | speedup vs picker |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    # Some sections live entirely under NEW (the 8b_b3b4b5 re-run was
    # a fresh full sweep, not a re-use of OLD baselines). Detect them by
    # the absence of an OLD/<model_tag>/ directory.
    baseline_root = OLD if (OLD / model_tag).exists() else NEW
    for tag, kind, fname, label in cells:
        old_path = baseline_root / model_tag / fname
        new_path = NEW / model_tag / fname
        names = BATCHED_NAMES if kind == "batched" else SGLANG_NAMES
        load = load_batched if kind == "batched" else load_sglang
        # Baselines + picker + 2ldt come from OLD picker_demo (or NEW
        # when the cell was run fresh). 2l1p_single always from NEW.
        values = {
            "paged":      load(old_path, names["paged"]),
            "fasttree":   load(old_path, names["fasttree"]),
            "deft":       load(old_path, names["deft"]),
            "bs_kernel":  load(old_path, names["bs_kernel"]),
            "2ldt":       load(new_path, names["2ldt"]),
            "2l1p_single": load(new_path, names["2l1p_single"]),
        }
        winner = best(values)
        cells_out = []
        for k in ["paged", "fasttree", "deft", "bs_kernel", "2ldt", "2l1p_single"]:
            s = fmt(values[k])
            if k == winner and s != "—":
                s = f"**{s}**"
            cells_out.append(s)
        sp_paged = speedup(values["paged"], values["2l1p_single"])
        sp_picker = speedup(values["bs_kernel"], values["2l1p_single"])
        lines.append(f"| {tag} | {label} | " + " | ".join(cells_out) + f" | {sp_paged} | {sp_picker} |")
    lines.append("")
    return lines


def main():
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    out = [
        "# End-to-end results — picker_demo cells (new single-launch 2l1p_single vs baselines)",
        "",
        "Baselines (`paged`, `fasttree`, `deft`, `bs_kernel` picker, `2ldt`) come from the original `paper-exp/picker_demo/` runs — the `single_launch=True` kernel patch only affects the `bs_kernel_2l1p_single` path, so the other methods' numbers are unchanged from those runs.",
        "",
        "`2l1p_single` is the forced `Strategy.SHARED_2L_1POOL` dispatched through `FusedMultiLevelCascadeAttentionWrapper(single_launch=True)`, which collapses the per-step cascade into one `fused_paged_run` launch.",
        "",
        "Speedup columns are `<other_method> / 2l1p_single` (×1 = tie, >1 = 2l1p_single faster).",
        "",
    ]
    for header, model_tag, cells in MODELS:
        out.extend(render_model(header, model_tag, cells, only=only))

    out.append("## Old vs new sanity check (rows shared between runs)")
    out.append("")
    out.append("`bs_kernel_2l1p` (multi-launch, in the old picker_demo) vs `bs_kernel_2l1p_single` (new) — both running `Strategy.SHARED_2L_1POOL`, identical model/dtype/prompts. Also `2ldt` re-run (should be ≈0 since the patch doesn't touch that path).")
    out.append("")
    out.append("| model | cell | OLD 2l1p | NEW 2l1p_single | Δ2l1p | OLD 2ldt | NEW 2ldt | Δ2ldt |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for header, model_tag, cells in MODELS:
        if only is not None and model_tag not in only:
            continue
        if not (OLD / model_tag).exists():
            # Fresh sweep — no OLD vs NEW to compute.
            continue
        for tag, kind, fname, label in cells:
            old_path = OLD / model_tag / fname
            new_path = NEW / model_tag / fname
            names = BATCHED_NAMES if kind == "batched" else SGLANG_NAMES
            load = load_batched if kind == "batched" else load_sglang
            old_2l1p_name = "bs_kernel_2l1p" if kind == "batched" else "bsk_2l_1p"
            op = load(old_path, old_2l1p_name)
            od = load(old_path, names["2ldt"])
            ns = load(new_path, names["2l1p_single"])
            nd = load(new_path, names["2ldt"])
            if op is None and ns is None and od is None and nd is None:
                continue
            def pct(o, n):
                if o is None or n is None or o == 0:
                    return "—"
                return f"{(n-o)/o*100:+.1f}%"
            out.append(f"| {model_tag} | {tag} | {fmt(op)} | {fmt(ns)} | {pct(op, ns)} | {fmt(od)} | {fmt(nd)} | {pct(od, nd)} |")
    out.append("")
    out.append("Δ convention: `(new − old) / old`. Positive = new is slower. The Δ2ldt column shows run-to-run variance only (≈±5% at sglang harness).")
    out.append("")

    Path("final_paper_e2e_table.md").write_text("\n".join(out))
    print(f"Wrote final_paper_e2e_table.md")


if __name__ == "__main__":
    main()

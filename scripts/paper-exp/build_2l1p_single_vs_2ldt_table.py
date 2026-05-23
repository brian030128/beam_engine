"""Generate final_paper_e2e_table.md comparing the OLD picker_demo
(bs_kernel_2l1p multi-launch + bs_kernel_2ldt) against the NEW
picker_demo_2l1p_single_vs_2ldt (bs_kernel_2l1p_single + bs_kernel_2ldt).

Cells and metrics mirror the picker_demo paper layout:
  X1 (batched, hotpotqa beam_search): decode_per_token_ms
  X2-XN (sglang scenarios): decode_total_ms, median over 3 repeats per method.
"""
from __future__ import annotations

import csv
import os
from statistics import median
from pathlib import Path

OLD = Path("benchmarks/bs_kernel/results/paper-exp/picker_demo")
NEW = Path("benchmarks/bs_kernel/results/paper-exp/picker_demo_2l1p_single_vs_2ldt")

MODELS = [
    # (header tag, dir tag, batched-method name in OLD CSV,
    #  cells: [(tag, kind, filename, label)])
    ("Llama-3.2-1B (bf16 w + bf16 KV)", "Llama-3.2-1B-bf16", [
        ("A1", "batched", "A1_beam_search_K4_B1_Lp40000_mn64.csv",   "beam_search K=4 B=1 L_p=40k mn=64"),
        ("A2", "sglang",  "A2_multi_chain_reasoning_K4_B1_mn32.csv", "multi_chain_reasoning K=4 B=1 mn=32"),
        ("A3", "sglang",  "A3_multi_doc_qa_K32_B1_mn256.csv",        "multi_doc_qa K=32 B=1 mn=256"),
        ("A4", "sglang",  "A4_multi_few_shot_K16_B8_mn256.csv",      "multi_few_shot K=16 B=8 mn=256"),
        ("A5", "sglang",  "A5_multi_level_system_K32_B4_mn256.csv",  "multi_level_system K=32 B=4 mn=256"),
    ]),
    ("Llama-3.1-8B (bf16 w + bf16 KV)", "Llama-3.1-8B-bf16", [
        ("B1", "batched", "B1_beam_search_K4_B1_Lp40000_mn128.csv",  "beam_search K=4 B=1 L_p=40k mn=128"),
        ("B2", "sglang",  "B2_multi_chain_reasoning_K4_B1_mn32.csv", "multi_chain_reasoning K=4 B=1 mn=32"),
        ("B3", "sglang",  "B3_multi_doc_qa_K64_B1_mn256.csv",        "multi_doc_qa K=64 B=1 mn=256"),
        ("B4", "sglang",  "B4_multi_few_shot_K32_B4_mn256.csv",      "multi_few_shot K=32 B=4 mn=256"),
        ("B5", "sglang",  "B5_multi_level_system_K64_B4_mn256.csv",  "multi_level_system K=64 B=4 mn=256"),
    ]),
    ("Llama-3-70B-Instruct-FP8 (TP=2, bf16 act + fp8_e4m3 KV)", "Llama-3-70B-Instruct-FP8", [
        ("C1", "batched", "C1_beam_search_K4_B1_Lp40000_mn64.csv",     "beam_search K=4 B=1 L_p=40k mn=64"),
        ("C2", "sglang",  "C2_multi_chain_reasoning_K4_B1_mn32.csv",   "multi_chain_reasoning K=4 B=1 mn=32"),
        ("C3", "sglang",  "C3_multi_doc_qa_K32_B1_mn256.csv",          "multi_doc_qa K=32 B=1 mn=256"),
        ("C4", "sglang",  "C4_multi_few_shot_K16_B4_mn256.csv",        "multi_few_shot K=16 B=4 mn=256"),
        ("C5", "sglang",  "C5_multi_level_system_K32_B4_mn256.csv",    "multi_level_system K=32 B=4 mn=256"),
        ("C6", "sglang",  "C6_multi_chain_reasoning_K64_B4_mn256.csv", "multi_chain_reasoning K=64 B=4 mn=256"),
        ("D1", "sglang",  "D1_multi_chain_reasoning_K32_B32_mn256.csv","multi_chain_reasoning K=32 B=32 mn=256"),
        ("D2", "sglang",  "D2_multi_few_shot_K16_B32_mn256.csv",       "multi_few_shot K=16 B=32 mn=256"),
        ("D3", "sglang",  "D3_multi_level_system_K32_B16_mn256.csv",   "multi_level_system K=32 B=16 mn=256"),
        ("D4", "sglang",  "D4_multi_doc_qa_K16_B4_mn256.csv",          "multi_doc_qa K=16 B=4 mn=256"),
    ]),
]


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


def fmt(x, width=10, prec=3):
    return f"{x:>{width}.{prec}f}" if x is not None else f"{'—':>{width}}"


def pct(old, new):
    if old is None or new is None or old == 0:
        return "—"
    delta = (new - old) / old * 100.0
    return f"{delta:+.1f}%"


def main() -> None:
    out = ["# 2l1p single-launch vs 2ldt — full picker_demo cells (old vs new)\n",
           "",
           "**OLD** = `bs_kernel_2l1p` (multi-launch) + `bs_kernel_2ldt` from the original `paper-exp/picker_demo/` runs.",
           "**NEW** = `bs_kernel_2l1p_single` (`FusedMultiLevelCascadeAttentionWrapper(single_launch=True)` → one `fused_paged_run` launch per step) + `bs_kernel_2ldt`, run with identical model/dtype/prompts.",
           "",
           "Metrics: cell `X1` reports `decode_per_token_ms` (batched harness, hotpotqa). All other cells report `decode_total_ms` median over 3 repeats (sglang harness).",
           "",
           "`Δ2l1p` is **new 2l1p_single vs old 2l1p**: negative = single-launch is slower than the old multi-launch path.",
           "`Δ2ldt` should be ≈ 0 — the patch is opt-in via `single_launch=True`, so the DEC_TAIL path is bit-identical to old; the gap is run-to-run variance.",
           "",
           ]
    for header, model_tag, cells in MODELS:
        out.append(f"## {header}\n")
        out.append("| cell | scenario / config | OLD 2l1p | OLD 2ldt | NEW 2l1p_single | NEW 2ldt | Δ2l1p | Δ2ldt |")
        out.append("|------|-------------------|---------:|---------:|----------------:|---------:|------:|------:|")
        for tag, kind, fname, label in cells:
            old_path = OLD / model_tag / fname
            new_path = NEW / model_tag / fname
            if kind == "batched":
                op = load_batched(old_path, "bs_kernel_2l1p")
                od = load_batched(old_path, "bs_kernel_2ldt")
                ns = load_batched(new_path, "bs_kernel_2l1p_single")
                nd = load_batched(new_path, "bs_kernel_2ldt")
            else:
                op = load_sglang(old_path, "bsk_2l_1p")
                od = load_sglang(old_path, "bsk_2l_dt")
                ns = load_sglang(new_path, "bsk_2l_1p_single")
                nd = load_sglang(new_path, "bsk_2l_dt")
            out.append(
                f"| {tag} | {label} | {fmt(op)} | {fmt(od)} | {fmt(ns)} | {fmt(nd)} | "
                f"{pct(op, ns)} | {pct(od, nd)} |"
            )
        out.append("")
    Path("final_paper_e2e_table.md").write_text("\n".join(out))
    print(f"Wrote final_paper_e2e_table.md ({sum(len(l) for l in out)} chars)")


if __name__ == "__main__":
    main()

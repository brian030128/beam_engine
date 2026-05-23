"""Regenerate final_paper_results.md (decode + plan-time tables) from the
canonical F1-F4 cell CSVs. Re-run after any backfill (e.g. mlca 70B).

Plan time = per-step index build + (for bs_kernel) the cost-model
dispatch decision, summed over decode steps (`plan_total_ms`). The
decision sub-component is `dispatch_total_ms` (sglang cells only),
reported in parens for bs_kernel.
"""
from __future__ import annotations
import csv, glob, os, re, statistics as s
from collections import defaultdict

BASE = "benchmarks/bs_kernel/results/paper-exp/final_paper"
MODELS = [
    ("Llama-3.2-1B-bf16", "Llama-3.2-1B (bf16 weights + bf16 KV, TP=1)"),
    ("Llama-3.1-8B-bf16", "Llama-3.1-8B (bf16 weights + bf16 KV, TP=1)"),
    ("Llama-3-70B-Instruct-FP8", "Llama-3-70B-Instruct-FP8 (fp8 weights + fp8_e4m3 KV, TP=2)"),
]
COLS = ["paged", "fasttree", "deft", "mlca", "bs_kernel", "1p", "dt"]
# On 70B, fasttree/deft have no fp8-KV support -> always "—".
DROP_70B = {"fasttree", "deft"}

CELLS = [
    ("F1 multi_doc_qa",    "K=128 B=1 ~80K", "F1_multi_doc_qa_K128_B1_mn256.csv"),
    ("F2 multi_few_shot",  "K=128 B={B}",    "F2_multi_few_shot_K128_B*_mn256.csv"),
    ("F3 beam_search",     "K=64 B={B} L_p=16K", "F3_beam_search_K64_B*_Lp16000_mn256.csv"),
    ("F4 self_consistency","K=16 B=1 L_p=16K","F4_self_consistency_K16_B1_Lp16384_mn256.csv"),
]

def norm(m):
    if m in ("bsk_2l_1p", "bsk_2l_1p_single", "bs_kernel_2l1p"): return "1p"
    if m in ("bsk_2l_dt", "bs_kernel_2ldt"): return "dt"
    return m

def core(tag):
    return {"paged","bs_kernel","1p","dt"} if tag.startswith("Llama-3-70B") \
        else {"paged","fasttree","deft","mlca","bs_kernel","1p","dt"}

def read(f, col):
    d = defaultdict(list)
    for r in csv.DictReader(open(f)):
        try: d[norm(r["method"])].append(float(r[col]))
        except (KeyError, ValueError): pass
    return {k: s.median(v) for k, v in d.items() if v}

def pick(tag, pat):
    """largest-B file whose core methods are all present (settled cell)."""
    best, bestB = None, -1
    for f in glob.glob(f"{BASE}/{tag}/{pat}"):
        if core(tag) <= set(read(f, "decode_total_ms")):
            B = int(re.search(r'_B(\d+)_', f).group(1)) if '_B' in f else 0
            if B > bestB: best, bestB = f, B
    return best

def fmt(m, val, win, drop, decmap=None):
    if m in drop or val is None or m not in val: return "—"
    v = f"{val[m]:.0f}"
    if decmap and m == "bs_kernel" and m in decmap: v += f" ({decmap[m]:.0f})"
    if m == win: v = f"**{v}**"
    return v

out = ["# Final paper — end-to-end results (F1–F4)", "",
       "Median over 3 warm repeats; lower = better. `1p`/`dt` = forced "
       "SHARED_2L_1POOL / SHARED_2L_DEC_TAIL; `bs_kernel` = cost-model picker. "
       "**bold** = row winner. F4 uses single-launch 1p + single-launch picker.", ""]

for tag, title in MODELS:
    drop = DROP_70B if tag.startswith("Llama-3-70B") else set()
    cells = []
    for name, cfg, pat in CELLS:
        f = f"{BASE}/{tag}/{pat}" if "*" not in pat else pick(tag, pat)
        B = re.search(r'_B(\d+)_', f).group(1) if f and '_B' in f else "1"
        cells.append((name, cfg.format(B=B), f))

    # ---- decode table ----
    out += [f"## {title}", "", "### Decode time (median `decode_total_ms`)", "",
            "| cell | config | " + " | ".join(COLS) + " | winner |",
            "|---|---|" + "---:|"*len(COLS) + "---|"]
    for name, cfg, f in cells:
        if not f or not os.path.exists(f):
            out.append(f"| {name} | {cfg} | " + " | ".join(["—"]*len(COLS)) + " | — |"); continue
        d = read(f, "decode_total_ms")
        valid = {k: v for k, v in d.items() if k not in drop}
        win = min(valid, key=valid.get) if valid else None
        out.append(f"| {name} | {cfg} | " +
                   " | ".join(fmt(m, d, win, drop) for m in COLS) + f" | {win} |")
    out.append("")

    # ---- plan table ----
    out += ["### Plan time (median `plan_total_ms`; bs_kernel shown as "
            "build-indices+decision, decision in parens)", "",
            "| cell | config | " + " | ".join(COLS) + " |",
            "|---|---|" + "---:|"*len(COLS)]
    for name, cfg, f in cells:
        if not f or not os.path.exists(f):
            out.append(f"| {name} | {cfg} | " + " | ".join(["—"]*len(COLS)) + " |"); continue
        p = read(f, "plan_total_ms")
        dec = read(f, "dispatch_total_ms")  # decision part (sglang only)
        out.append(f"| {name} | {cfg} | " +
                   " | ".join(fmt(m, p, None, drop, dec) for m in COLS) + " |")
    out.append("")

out += ["## Notes", "",
    "- **Scenarios.** F1 multi_doc_qa (~80K shared prefix, B=1, K=128); F2 multi_few_shot (~4K prefix, large B, K=128); F3 beam search (multi-level dynamic tree, L_p=16K, K=64); F4 self_consistency / best-of-N (static 2-level: shared prefix → K independent tails, B=1, K=16, L_p=16K).",
    "- **Max-fit batch size.** F2/F3 batch sizes are the largest where all methods fit one node; they differ by model (footnoted in the config column).",
    "- **fp8 baselines on 70B.** `fasttree`/`deft` have no fp8-KV support (no `kv_dtype` path) → shown as “—” on all 70B cells. `mlca` *does* support fp8 once `MlcaBackend.plan` is given the page-table store dtype (fixed in baselines/mlca.py).",
    "- **Plan time.** Sum of per-step index/metadata build over all decode steps. For `bs_kernel` it also includes the cost-model dispatch decision (shown in parens); the decision is a small fraction of bs_kernel's already-small plan cost, and bs_kernel's total plan is far below paged's index-build on the long-prefix cells.",
    ""]

open("final_paper_results.md", "w").write("\n".join(out))
print("wrote final_paper_results.md")

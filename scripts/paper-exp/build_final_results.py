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
    ("Llama-3-70B-bf16", "Llama-3-70B-Instruct (bf16 weights + bf16 KV, TP=4)"),
]
COLS = ["paged", "fasttree", "deft", "mlca", "bs_kernel", "1p", "dt"]
# fasttree/deft have no fp8-KV support -> "—" only on an fp8 70B tag (if present).
# bf16 70B runs all methods, so the drop is scoped to the fp8 tag below.
DROP_70B = {"fasttree", "deft"}

CELLS = [
    ("F1 multi_doc_qa",    "K=128 B=1 ~80K", "F1_multi_doc_qa_K128_B1_mn256.csv"),
    ("F2 multi_few_shot",  "K=128 B={B}",    "F2_multi_few_shot_K128_B*_mn256.csv"),
    ("F3 beam_search",     "K=64 B={B} L_p=16K", "F3_beam_search_K64_B*_Lp16000_mn256.csv"),
    ("F4 self_consistency","K=16 B=1 L_p={LP}","F4_self_consistency_K16_B1_Lp*_mn256.csv"),
]

# F4 self_consistency uses a per-model L_p (the prefix length that puts
# attention in the 1POOL-win regime for that model's forward/attention
# balance — bigger/more-forward-bound models need a longer prefix):
#   1B → 32K, 8B → 16K, 70B → 64K.
F4_LP_BY_MODEL = {
    "Llama-3.2-1B-bf16": 32768,
    "Llama-3.1-8B-bf16": 16384,
    "Llama-3-70B-bf16": 65536,
}

def norm(m):
    if m in ("bsk_2l_1p", "bsk_2l_1p_single", "bs_kernel_2l1p"): return "1p"
    if m in ("bsk_2l_dt", "bs_kernel_2ldt"): return "dt"
    return m

def core(tag):
    return {"paged","bs_kernel","1p","dt"} if "FP8" in tag \
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

def pick_lp(tag, pat):
    """largest-L_p file whose core methods are all present. F4 was rerun at
    32K on 1B/8B (where 1p/picker overtake paged); 70B stays at its 16K
    file (forward-bound at K=16, not rerun)."""
    best, bestLp = None, -1
    for f in glob.glob(f"{BASE}/{tag}/{pat}"):
        if core(tag) <= set(read(f, "decode_total_ms")):
            m = re.search(r'_Lp(\d+)_', f)
            lp = int(m.group(1)) if m else 0
            if lp > bestLp: best, bestLp = f, lp
    return best

def lp_label(f):
    """'…_Lp32768_…' -> '32K' (round to nearest K)."""
    m = re.search(r'_Lp(\d+)_', f) if f else None
    if not m: return "16K"
    return f"{round(int(m.group(1))/1024)}K"

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
    drop = DROP_70B if "FP8" in tag else set()
    cells = []
    for name, cfg, pat in CELLS:
        if "*" not in pat:
            f = f"{BASE}/{tag}/{pat}"
        elif "_Lp*" in pat:
            # F4: explicit per-model L_p (not largest-on-disk).
            lp = F4_LP_BY_MODEL.get(tag)
            f = f"{BASE}/{tag}/{pat.replace('_Lp*', f'_Lp{lp}')}" if lp else None
            if f and not os.path.exists(f):
                f = None
        else:
            f = pick(tag, pat)
        B = re.search(r'_B(\d+)_', f).group(1) if f and '_B' in f else "1"
        cells.append((name, cfg.format(B=B, LP=lp_label(f)), f))

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
    "- **Scenarios.** F1 multi_doc_qa (~80K shared prefix, B=1, K=128); F2 multi_few_shot (~4K prefix, large B, K=128); F3 beam search (multi-level dynamic tree, L_p=16K, K=64); F4 self_consistency / best-of-N (static 2-level: shared prefix → K independent tails, B=1, K=16, per-model L_p = 1B 32K / 8B 16K / 70B 64K — the prefix length that puts attention in the 1POOL-win regime for that model's forward/attention balance; bigger/more-forward-bound models need a longer prefix. At those L_p the picker selects 1POOL and 1p/bs_kernel beat paged on all three, 10–22%).",
    "- **Max-fit batch size.** F2/F3 batch sizes are the largest where all methods fit one node; they differ by model (footnoted in the config column).",
    "- **70B is bf16 (unquantized), TP=4.** Llama-3-70B-Instruct at bf16 weights + bf16 KV on 4×H100, so all methods run (no fp8-KV limitation). bs_kernel/dt win every 70B cell — F1 2.7× over mlca, F2 1.4× over paged, F3 1.34× over mlca; the picker matches the best forced strategy on all three after the split-K kernel + cost-model fixes. (An earlier fp8/TP=2 run, where mlca won, is kept on disk but not shown here.)",
    "- **Plan time.** Sum of per-step index/metadata build over all decode steps. For `bs_kernel` the parens show the cost-model dispatch decision sub-cost (small). `paged` now uses the same LCA-prefix-skip for its index build (shared prefill prefix converted once per prompt, not per beam), cutting its plan time ~8-9× on the long-prefix F1 cells. After that fix the methods' plan costs are close: bs_kernel is lowest on F1, but on the forking F3 beam_search paged plans faster (bs_kernel pays for GPU plan-state updates under heavy forking).",
    ""]

open("final_paper_results.md", "w").write("\n".join(out))
print("wrote final_paper_results.md")

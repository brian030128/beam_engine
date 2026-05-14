#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_warm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_paper_exact_warm_%j.out
#SBATCH --error=slurm/logs/bench_paper_exact_warm_%j.err

# Paper-exact rerun with --warmup --repeat 3 to discard the first
# (Triton-JIT autotune) iteration and median over 3 warm runs.
#
# Two sub-invocations so chain_reasoning uses K=16 while the other two
# scenarios keep their natural K:
#   pass 1 — multi_level_system + multi_few_shot  (natural K)
#   pass 2 — multi_chain_reasoning at --k-override 16
#
# 3 methods × 2 models. Median of 3 warm iterations.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/paper_exact_warm-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== paper_exact_warm started $(date -Iseconds) ==="
    echo "scenarios: multi_level_system multi_few_shot (natural K)"
    echo "         + multi_chain_reasoning at K=16"
    echo "methods:  fasttree bs_kernel paged"
    echo "models:   Llama-3.2-1B, Llama-3.1-8B"
    echo "--warmup --repeat 3 --max-new 256"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    export BE_MODEL="meta-llama/${model_tag}"
    csv_a="$RESULTS_DIR/${model_tag}_a.csv"
    csv_b="$RESULTS_DIR/${model_tag}_b.csv"
    echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"
    echo "  pass 1: level_system + few_shot (natural K)" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_level_system multi_few_shot \
        --methods fasttree bs_kernel paged \
        --paper-exact \
        --max-new 256 --repeat 3 --warmup \
        --out "$csv_a" \
        2>&1 | tee -a "$LOG"
    echo
    echo "  pass 2: chain_reasoning at K=16" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_chain_reasoning \
        --methods fasttree bs_kernel paged \
        --paper-exact --k-override 16 \
        --max-new 256 --repeat 3 --warmup \
        --out "$csv_b" \
        2>&1 | tee -a "$LOG"
    # Merge a+b → ${model_tag}.csv (header from a, drop header from b).
    merged="$RESULTS_DIR/${model_tag}.csv"
    cp "$csv_a" "$merged"
    tail -n +2 "$csv_b" >> "$merged"
    echo "  merged → $merged" | tee -a "$LOG"
    echo | tee -a "$LOG"
done

# Median-of-3 paper table.
PAPER="$RESULTS_DIR/paper_table.txt"
uv run python - <<PY 2>&1 | tee "$PAPER"
import csv, statistics
from pathlib import Path
dir_ = Path("$RESULTS_DIR")
all_rows = []
for f in sorted(dir_.glob("Llama-*.csv")):
    if "_a.csv" in f.name or "_b.csv" in f.name:
        continue
    model = f.stem
    with f.open() as fh:
        for r in csv.DictReader(fh):
            r["model"] = model
            all_rows.append(r)

def med(rows, key):
    vals = [float(r[key]) for r in rows]
    return statistics.median(vals) if vals else 0.0

print("\n========== FastTree paper-exact (warmup, median of 3) ==========")
print(f"{'model':<14} {'scenario':<24} {'method':<10} "
      f"{'decode_total':>12} {'plan_total':>11} {'fwd_total':>11} "
      f"{'plan%':>6} {'fwd%':>6} {'p50/step':>9}")
for model in sorted({r["model"] for r in all_rows}):
    for scen in ("multi_level_system","multi_few_shot","multi_chain_reasoning"):
        for meth in ("fasttree","bs_kernel","paged"):
            cells = [r for r in all_rows
                     if r["model"]==model and r["scenario"]==scen
                     and r["method"]==meth]
            if not cells: continue
            dt = med(cells, "decode_total_ms")
            pt = med(cells, "plan_total_ms")
            ft = med(cells, "forward_total_ms")
            p50 = med(cells, "decode_p50_ms")
            pct_p = 100.0 * pt / dt if dt > 0 else 0.0
            pct_f = 100.0 * ft / dt if dt > 0 else 0.0
            tag = scen if scen != "multi_chain_reasoning" else "multi_chain_reasoning(K=16)"
            print(f"{model:<14} {tag:<24} {meth:<10} "
                  f"{dt:>12.1f} {pt:>11.1f} {ft:>11.1f} "
                  f"{pct_p:>5.1f}% {pct_f:>5.1f}% {p50:>9.2f}")
print(f"\nfull CSVs: $RESULTS_DIR")
PY

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

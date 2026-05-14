#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=phase1_200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_phase1_200data_%j.out
#SBATCH --error=slurm/logs/bench_phase1_200data_%j.err

# K=16 (FastTree phase 1) × B=13 (~208 leaves per scenario) ×
# 4 methods × 2 models, all 4 scenarios. ~200 distinct GSM8K
# questions per scenario for the three fan-out-at-prefill cases;
# for multi_chain_reasoning, 13 questions × 16 chains = 208 leaves
# but only 13 unique questions (K chains per question is intrinsic).

set -uo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/phase1_200data-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== phase1 200-data sweep started $(date -Iseconds) ==="
    echo "K=16  B=13  scenarios: multi_level_system multi_few_shot multi_chain_reasoning multi_document"
    echo "methods: fasttree bs_kernel paged mlca"
    echo "models:  Llama-3.2-1B, Llama-3.1-8B"
    echo "max_new=256, repeat=1"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    export BE_MODEL="meta-llama/${model_tag}"
    csv="$RESULTS_DIR/${model_tag}.csv"
    echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_level_system multi_few_shot \
                    multi_chain_reasoning multi_document \
        --methods fasttree bs_kernel paged mlca \
        --k-override 16 \
        --b-override 13 \
        --max-new 256 \
        --repeat 1 \
        --out "$csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

# Paper table.
PAPER="$RESULTS_DIR/paper_table.txt"
uv run python - <<PY 2>&1 | tee "$PAPER"
import csv
from pathlib import Path
dir_ = Path("$RESULTS_DIR")
files = sorted(dir_.glob("Llama-*.csv"))
rows = []
for f in files:
    model = f.stem
    with f.open() as fh:
        for r in csv.DictReader(fh):
            r["model"] = model
            rows.append(r)
def fmt(x): return f"{float(x):7.1f}"

print("\n========== K=16 (phase 1)  ·  B=13 (~208 leaves / scenario) ==========")
print(f"{'model':<14} {'scenario':<24} {'method':<10} "
      f"{'decode_total':>12} {'plan_total':>11} {'fwd_total':>11} "
      f"{'plan%':>6} {'fwd%':>6} {'p50/step':>9}")
for model in sorted({r["model"] for r in rows}):
    for scen in ("multi_level_system","multi_few_shot",
                 "multi_chain_reasoning","multi_document"):
        for meth in ("fasttree","bs_kernel","paged","mlca"):
            cell = [r for r in rows
                    if r["model"]==model and r["scenario"]==scen
                    and r["method"]==meth]
            if not cell: continue
            r = cell[0]
            dt = float(r["decode_total_ms"]); pt = float(r["plan_total_ms"])
            ft = float(r["forward_total_ms"]); p50 = float(r["decode_p50_ms"])
            pct_p = 100.0 * pt / dt if dt > 0 else 0.0
            pct_f = 100.0 * ft / dt if dt > 0 else 0.0
            print(f"{model:<14} {scen:<24} {meth:<10} "
                  f"{fmt(dt):>12} {fmt(pt):>11} {fmt(ft):>11} "
                  f"{pct_p:>5.1f}% {pct_f:>5.1f}% {p50:>9.2f}")
print(f"\nfull CSVs: $RESULTS_DIR")
PY

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=chain_K
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_chain_K_sweep_%j.out
#SBATCH --error=slurm/logs/bench_chain_K_sweep_%j.err

# Paper-exact multi_chain_reasoning at larger K (4 chains → 8/16).
# Same B=32, two-stage fork+merge; only K varies.
# 3 methods × 3 K values × 2 models = 18 cells.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/chain_K_sweep-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== chain_K_sweep started $(date -Iseconds) ==="
    echo "scenario: multi_chain_reasoning (paper-exact two-stage)"
    echo "K values: 4 (paper default), 8, 16"
    echo "methods:  fasttree bs_kernel paged"
    echo "models:   Llama-3.2-1B, Llama-3.1-8B"
    echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    export BE_MODEL="meta-llama/${model_tag}"
    csv="$RESULTS_DIR/${model_tag}.csv"
    echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_chain_reasoning \
        --methods fasttree bs_kernel paged \
        --paper-exact \
        --k-override 4,8,16 \
        --max-new 256 --repeat 1 \
        --out "$csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

PAPER="$RESULTS_DIR/paper_table.txt"
uv run python - <<PY 2>&1 | tee "$PAPER"
import csv
from pathlib import Path
dir_ = Path("$RESULTS_DIR")
rows = []
for f in sorted(dir_.glob("Llama-*.csv")):
    model = f.stem
    with f.open() as fh:
        for r in csv.DictReader(fh):
            r["model"] = model
            rows.append(r)
def fmt(x): return f"{float(x):8.1f}"
print("\n========== multi_chain_reasoning paper-exact two-stage, K sweep ==========")
print(f"{'model':<14} {'K':<3} {'method':<10} "
      f"{'decode_total':>13} {'plan_total':>11} {'fwd_total':>11} "
      f"{'plan%':>6} {'fwd%':>6} {'p50/step':>9}")
for model in sorted({r["model"] for r in rows}):
    for K in ("4", "8", "16"):
        for meth in ("fasttree","bs_kernel","paged"):
            cell = [r for r in rows
                    if r["model"]==model and r["k_override"]==K
                    and r["method"]==meth]
            if not cell: continue
            r = cell[0]
            dt = float(r["decode_total_ms"]); pt = float(r["plan_total_ms"])
            ft = float(r["forward_total_ms"]); p50 = float(r["decode_p50_ms"])
            pct_p = 100.0 * pt / dt if dt > 0 else 0.0
            pct_f = 100.0 * ft / dt if dt > 0 else 0.0
            print(f"{model:<14} {K:<3} {meth:<10} "
                  f"{fmt(dt):>13} {fmt(pt):>11} {fmt(ft):>11} "
                  f"{pct_p:>5.1f}% {pct_f:>5.1f}% {p50:>9.2f}")
print(f"\nfull CSVs: $RESULTS_DIR")
PY

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

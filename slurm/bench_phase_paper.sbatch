#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=phase_pap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_phase_paper_%j.out
#SBATCH --error=slurm/logs/bench_phase_paper_%j.err

# Paper-quality phase-0 vs phase-1 sweep with plan/forward decode-time
# breakdown. Two K-variants (K=16 → phase 1, K=32 → phase 0) × 4
# scenarios × 4 methods × 2 models. OOM-tolerant.
#
# Per-cell decode_total / plan_* / forward_* / alloc_* / topk_* are all
# written to CSV.  At the end, a tiny inline Python snippet prints a
# paper-ready table joined across both models.

set -uo pipefail   # no -e: a single OOM must not abort the loop

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/phase_paper-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== bench_phase_paper started $(date -Iseconds) ==="
    echo "scenarios: multi_level_system multi_few_shot multi_chain_reasoning multi_document"
    echo "methods:   fasttree bs_kernel paged mlca"
    echo "K vars:    16 (phase 1)  /  32 (phase 0)"
    echo "max_new:   256, repeat: 1, no warmup"
    echo "models:    Llama-3.2-1B  /  Llama-3.1-8B"
    echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
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
        --k-override 16,32 \
        --max-new 256 \
        --repeat 1 \
        --out "$csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

# Paper table: per K-variant, scenario × method → decode_total / plan_total / forward_total / plan%.
PAPER="$RESULTS_DIR/paper_table.txt"
uv run python - <<PY 2>&1 | tee "$PAPER"
import csv, sys
from pathlib import Path

dir_ = Path("$RESULTS_DIR")
files = sorted(dir_.glob("Llama-*.csv"))
if not files:
    print("no CSVs found")
    sys.exit(0)

rows = []
for f in files:
    model = f.stem  # "Llama-3.2-1B" etc.
    with f.open() as fh:
        for r in csv.DictReader(fh):
            r["model"] = model
            rows.append(r)

def fmt(x): return f"{float(x):7.1f}"

for k_override in ("16", "32"):
    phase = "phase 1" if k_override == "16" else "phase 0"
    print(f"\n========== K={k_override}  ({phase}) ==========")
    print(f"{'model':<14} {'scenario':<24} {'method':<10} "
          f"{'decode_total':>12} {'plan_total':>11} {'fwd_total':>11} "
          f"{'plan%':>6} {'fwd%':>6} {'p50/step':>9}")
    for model in sorted({r["model"] for r in rows}):
        for scen in ("multi_level_system","multi_few_shot",
                     "multi_chain_reasoning","multi_document"):
            for meth in ("fasttree","bs_kernel","paged","mlca"):
                cell = [r for r in rows
                        if r["model"]==model
                        and r["k_override"]==k_override
                        and r["scenario"]==scen
                        and r["method"]==meth]
                if not cell: continue
                r = cell[0]
                dt = float(r["decode_total_ms"])
                pt = float(r["plan_total_ms"])
                ft = float(r["forward_total_ms"])
                p50 = float(r["decode_p50_ms"])
                pct_p = 100.0 * pt / dt if dt > 0 else 0.0
                pct_f = 100.0 * ft / dt if dt > 0 else 0.0
                print(f"{model:<14} {scen:<24} {meth:<10} "
                      f"{fmt(dt):>12} {fmt(pt):>11} {fmt(ft):>11} "
                      f"{pct_p:>5.1f}% {pct_f:>5.1f}% {p50:>9.2f}")

print(f"\nfull CSVs: $RESULTS_DIR")
PY

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

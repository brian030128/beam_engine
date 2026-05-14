#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=smk_phase
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/smoke_phase_paper_%j.out
#SBATCH --error=slurm/logs/smoke_phase_paper_%j.err

set -uo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/smoke_phase-${TS}"
mkdir -p "$RESULTS_DIR"

echo "=== smoke: phase-timing plumbing + K=16/32 override ==="
# Tiny config: 1 scenario, 1 method, max_new=16.
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_few_shot --methods fasttree \
    --k-override 16,32 \
    --max-new 16 --repeat 1 \
    --out "$RESULTS_DIR/smoke.csv"

echo
echo "=== CSV content (check 12 new columns + phase totals sum ≈ decode_total) ==="
cat "$RESULTS_DIR/smoke.csv"

echo
echo "=== sanity sum: plan_total + forward_total + alloc_total + topk_total vs decode_total ==="
uv run python - <<'PY'
import csv
from pathlib import Path
import glob
csv_path = sorted(glob.glob("benchmarks/bs_kernel/results/smoke_phase-*/smoke.csv"))[-1]
with open(csv_path) as f:
    for r in csv.DictReader(f):
        dt = float(r["decode_total_ms"])
        s = sum(float(r[k]) for k in
                ("plan_total_ms","forward_total_ms","alloc_total_ms","topk_total_ms"))
        ratio = s/dt if dt>0 else float("nan")
        print(f"  {r['scenario']:<24} K={r['k_override']:>2}  "
              f"decode_total={dt:7.2f}  sum_phases={s:7.2f}  ratio={ratio:.3f}")
PY
echo "Done."

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=diag_sync
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:08:00
#SBATCH --output=slurm/logs/diag_sync_%j.out
#SBATCH --error=slurm/logs/diag_sync_%j.err

# Re-run the K=64/L_p=8192/B=32/max_new=256 'auto' mode with two extra
# CUDA syncs added in driver.py:
#   1. post_forward_sync: a sync immediately after mark("forward_ms")
#      — if its time is ~0, the existing forward sync was sufficient
#      and beam_select's ~40 ms is real Python overhead, not stale
#      forward-kernel wait.
#   2. an extra sync before mark("beam_select_ms") so any per-prompt
#      .tolist() / topk pending work is attributed to beam_select.

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/diag_sync_check-${TS}"
mkdir -p "$RESULTS_DIR"

uv run python benchmarks/bs_kernel/diagnose_cell.py run \
    --mode auto \
    --out "$RESULTS_DIR/auto.json"

echo
echo "=== phase summary (with new post_forward_sync phase) ==="
uv run python benchmarks/bs_kernel/diagnose_cell.py phase_summary \
    --input "$RESULTS_DIR/auto.json"

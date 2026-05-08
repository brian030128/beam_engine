#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_modes_single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/bench_modes_single_%j.out
#SBATCH --error=slurm/logs/bench_modes_single_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_modes_single-${TS}"
mkdir -p "$RESULTS_DIR"

# Single grid point: K=64 × L_p=8192 × B=32 × max_new=256
# All 6 bs_kernel modes (picker auto + 5 forced strategies).
# SHARED_3L_* will probably fail at the first step (no intermediate
# groups right after prefill); that's expected.
uv run python benchmarks/bs_kernel/bench_modes.py \
    --K 64 \
    --L_p 8192 \
    --B 32 \
    --max_new 256 \
    --out "$RESULTS_DIR/bench_modes_single.csv" \
    2>&1 | tee "$RESULTS_DIR/bench_modes_single.log"

echo
echo "Done. Results: $RESULTS_DIR"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_dispatch_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_dispatch_grid_%j.out
#SBATCH --error=slurm/logs/bench_dispatch_grid_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_dispatch_grid-${TS}"
mkdir -p "$RESULTS_DIR"

# Map empirical Pareto frontier across cells.
# Vanilla beam search (identical-prompt batches, no intermediate group structure
# → SHARED_3L_* and DEC_TAIL behave like SHARED_2L_* with merge fallbacks).
# Sweeps:
#   K ∈ {16, 64}       — small vs large beam count
#   L_p ∈ {2048, 8192, 32768}   — short / mid / long prefix
#   B ∈ {1, 8, 32}     — single, mid, large batch
#   max_new=128        — moderate decode length
# All 8 modes per cell.
uv run python benchmarks/bs_kernel/bench_modes.py \
    --K 16 64 \
    --L_p 2048 8192 32768 \
    --B 1 8 32 \
    --max_new 128 \
    --out "$RESULTS_DIR/bench_modes.csv" \
    2>&1 | tee "$RESULTS_DIR/bench_modes.log"

echo
echo "Done. Results: $RESULTS_DIR"

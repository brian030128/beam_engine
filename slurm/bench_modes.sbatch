#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_modes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_modes_%j.out
#SBATCH --error=slurm/logs/bench_modes_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_modes-${TS}"
mkdir -p "$RESULTS_DIR"

# K {16, 64} x L_p=8192 x B {1, 16, 32} x 6 modes x max_new=512 = 36 runs
uv run python benchmarks/bs_kernel/bench_modes.py \
    --K 16 64 \
    --L_p 8192 \
    --B 1 16 32 \
    --max_new 512 \
    --out "$RESULTS_DIR/bench_modes.csv" \
    2>&1 | tee "$RESULTS_DIR/bench_modes.log"

echo
echo "Done. Results: $RESULTS_DIR"

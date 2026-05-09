#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_b32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_batched_b32_%j.out
#SBATCH --error=slurm/logs/bench_batched_b32_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_batched_b32-${TS}"
mkdir -p "$RESULTS_DIR"

uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 16 32 64 \
    --L_p 8192 \
    --B 1 2 4 8 16 32 \
    --max_new 16 \
    --methods paged tree fasttree mlca adaptive_pool bs_kernel \
    --out "$RESULTS_DIR/bench_batched_b32.csv" \
    2>&1 | tee "$RESULTS_DIR/bench_batched_b32.log"

echo
echo "Done. Results: $RESULTS_DIR"

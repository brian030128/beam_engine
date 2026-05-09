#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_k64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_long_decode_k64_%j.out
#SBATCH --error=slurm/logs/bench_long_decode_k64_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# Reduce allocator fragmentation across cells; mitigates the cumulative
# OOM that blew up the prior bench's K=16/L_p=50k/B=32 cell.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_long_decode_k64-${TS}"
mkdir -p "$RESULTS_DIR"

uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 64 \
    --L_p 8192 50000 \
    --B 1 16 32 \
    --max_new 512 \
    --methods paged tree fasttree mlca adaptive_pool bs_kernel \
    --out "$RESULTS_DIR/bench_long_decode_k64.csv" \
    2>&1 | tee "$RESULTS_DIR/bench_long_decode_k64.log"

echo
echo "Done. Results: $RESULTS_DIR"

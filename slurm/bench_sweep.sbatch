#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/bench_sweep_%j.out
#SBATCH --error=slurm/logs/bench_sweep_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_sweep-${TS}"
mkdir -p "$RESULTS_DIR"

# Cross-method workload sweep at L_p=8192 / max_new=256.
# Cells: (K=4, B=4) small | (K=16, B=16) medium | (K=64, B=32) large.
# Methods: paged, mlca, adaptive_pool, fasttree, tree, bs_kernel (ours).
# tree/fasttree are slow (custom Triton kernels with no plan caching);
# they go last so they don't block faster methods if SLURM budget is hit.
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 4 16 64 \
    --L_p 8192 \
    --B 4 16 32 \
    --max_new 256 \
    --methods paged mlca adaptive_pool bs_kernel fasttree tree \
    --out "$RESULTS_DIR/sweep.csv" \
    2>&1 | tee "$RESULTS_DIR/sweep.log"

echo
echo "Done. Results: $RESULTS_DIR"

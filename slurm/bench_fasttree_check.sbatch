#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=ft_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/bench_fasttree_check_%j.out
#SBATCH --error=slurm/logs/bench_fasttree_check_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_fasttree_check-${TS}"
mkdir -p "$RESULTS_DIR"

# fasttree alone after the contiguous() bug fix at the canonical cell.
# Compare against bs_kernel-picker (post-DEC_TAIL) for context.
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 64 \
    --L_p 8192 \
    --B 32 \
    --max_new 256 \
    --methods fasttree bs_kernel \
    --out "$RESULTS_DIR/result.csv" \
    2>&1 | tee "$RESULTS_DIR/result.log"

echo
echo "Done. Results: $RESULTS_DIR"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bb_ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/bench_batched_ab_cache_%j.out
#SBATCH --error=slurm/logs/bench_batched_ab_cache_%j.err

set -euo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "######## A: decomp cache ENABLED (default) ########"
unset BS_KERNEL_DISABLE_DECOMP_CACHE
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 16 32 64 \
    --L_p 8192 \
    --B 1 2 4 8 \
    --max_new 16 \
    --methods bs_kernel paged

echo
echo "######## B: decomp cache DISABLED (baseline) ########"
export BS_KERNEL_DISABLE_DECOMP_CACHE=1
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 16 32 64 \
    --L_p 8192 \
    --B 1 2 4 8 \
    --max_new 16 \
    --methods bs_kernel paged

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=tree_kern
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/bench_tree_kernel_%j.out
#SBATCH --error=slurm/logs/bench_tree_kernel_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python benchmarks/bs_kernel/bench_tree_kernel.py \
    --csv benchmarks/bs_kernel/results/bench_tree_kernel-$(date +%Y%m%d-%H%M%S).csv

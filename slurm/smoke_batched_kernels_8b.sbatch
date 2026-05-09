#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=smoke_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/smoke_batched_kernels_8b_%j.out
#SBATCH --error=slurm/logs/smoke_batched_kernels_8b_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export BE_MODEL=meta-llama/Llama-3.1-8B
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python benchmarks/bs_kernel/smoke_batched_kernels.py

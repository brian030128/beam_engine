#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bsk_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/profile_bs_kernel_breakdown_8b_%j.out
#SBATCH --error=slurm/logs/profile_bs_kernel_breakdown_8b_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export BE_MODEL=meta-llama/Llama-3.1-8B
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_PROFILE_ATTN=1
export BE_K=64
export BE_L_P=8192
export BE_B=8
export BE_MAX_NEW=256

uv run python benchmarks/bs_kernel/profile_bs_kernel_breakdown.py

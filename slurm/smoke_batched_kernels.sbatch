#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=smoke_kernels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/smoke_batched_kernels_%j.out
#SBATCH --error=slurm/logs/smoke_batched_kernels_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

uv run python benchmarks/bs_kernel/smoke_batched_kernels.py

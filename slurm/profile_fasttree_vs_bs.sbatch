#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=prof_ft_bs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/profile_fasttree_vs_bs_%j.out
#SBATCH --error=slurm/logs/profile_fasttree_vs_bs_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python benchmarks/bs_kernel/profile_fasttree_vs_bs.py \
    --K 64 \
    --L_p 8192 \
    --B 32 \
    --max_new 256 \
    --methods fasttree bs_kernel

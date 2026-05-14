#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=sg_ftpl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/profile_ft_plan_%j.out
#SBATCH --error=slurm/logs/profile_ft_plan_%j.err

set -euo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FT_TRACE_PLAN=1

uv run python benchmarks/bs_kernel/profile_sglang_e2e.py \
    --scenario multi_chain_reasoning \
    --methods fasttree \
    --max-new 256

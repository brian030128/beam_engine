#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=sg_prof
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/profile_sglang_e2e_%j.out
#SBATCH --error=slurm/logs/profile_sglang_e2e_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1

echo "===== multi_chain_reasoning ====="
uv run python benchmarks/bs_kernel/profile_sglang_e2e.py \
    --scenario multi_chain_reasoning \
    --methods bs_kernel paged fasttree \
    --max-new 256

echo
echo "===== multi_document ====="
uv run python benchmarks/bs_kernel/profile_sglang_e2e.py \
    --scenario multi_document \
    --methods bs_kernel paged fasttree \
    --max-new 256

echo
echo "===== multi_level_system ====="
uv run python benchmarks/bs_kernel/profile_sglang_e2e.py \
    --scenario multi_level_system \
    --methods bs_kernel paged fasttree \
    --max-new 256

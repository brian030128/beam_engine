#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=mlca_fix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/mlca_fix_%j.out
#SBATCH --error=slurm/logs/mlca_fix_%j.err

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="meta-llama/Llama-3.2-1B"

# Quick test: mlca on multi_chain_reasoning with K=64 override (the failing
# case from exp1a). Short max_new to keep it fast.
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods mlca \
    --paper-exact \
    --scenarios multi_chain_reasoning \
    --k-override 64 --b-override 32 \
    --max-new 32 --repeat 1

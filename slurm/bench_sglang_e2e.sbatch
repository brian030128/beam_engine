#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=sg_e2e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/bench_sglang_e2e_%j.out
#SBATCH --error=slurm/logs/bench_sglang_e2e_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_level_system multi_few_shot multi_chain_reasoning multi_document \
    --methods bs_kernel fasttree paged mlca \
    --max-new 256 \
    --repeat 3 \
    --warmup

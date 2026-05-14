#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bsk_nocache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/test_bs_kernel_no_cache_%j.out
#SBATCH --error=slurm/logs/test_bs_kernel_no_cache_%j.err

set -euo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_TEST_MODEL=meta-llama/Llama-3.2-1B
export BS_KERNEL_DISABLE_DECOMP_CACHE=1

uv run python tests/test_bs_kernel_cache.py

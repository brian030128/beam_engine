#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=test_bsk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/test_bs_kernel_%j.out
#SBATCH --error=slurm/logs/test_bs_kernel_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cross-baseline correctness: bs_kernel vs tree.py across K, L_p, max_new.
# Validates the new SHARED-branch level-0 pack cache + plan_info cache
# do not change outputs.
#
# Use Llama-3.2-1B (head_dim=64) instead of the test's default 3.1-8B —
# the SM90 ragged-prefill kernel for head_dim=128 fails with
# "operation not supported" in this env, which is unrelated to our
# changes (same failure on baseline tree.py without any bs_kernel
# changes).
export BE_TEST_MODEL="meta-llama/Llama-3.2-1B"
uv run python tests/test_bs_kernel.py

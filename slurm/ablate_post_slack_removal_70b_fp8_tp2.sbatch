#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=ab_post_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/ablate_post_slack_removal_70b_fp8_tp2_%j.out
#SBATCH --error=slurm/logs/ablate_post_slack_removal_70b_fp8_tp2_%j.err

set -euo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_DTYPE=bf16

uv run torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/ablate_autotune_knobs.py \
    --model RedHatAI/Meta-Llama-3-70B-Instruct-FP8 \
    --dtype bf16 \
    --max-new 16

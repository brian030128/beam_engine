#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=ablate_70b_fp8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/ablate_autotune_knobs_70b_fp8_tp2_%j.out
#SBATCH --error=slurm/logs/ablate_autotune_knobs_70b_fp8_tp2_%j.err

set -euo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# 70B fp8 doesn't fit on /work — use SLURM_TMPDIR ephemeral cache.
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_DTYPE=bf16

echo "=== GPUs ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo

# max_new=16 matches the production autotune_h100_70b_fp8_tp2.sbatch:
# 70B's larger per-page slab + 80 layers makes the long-decode cells
# OOM, and the production fit used 16 successfully.
uv run torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/ablate_autotune_knobs.py \
    --model RedHatAI/Meta-Llama-3-70B-Instruct-FP8 \
    --dtype bf16 \
    --max-new 16

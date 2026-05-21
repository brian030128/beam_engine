#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=70b_fp8kv_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/70b_fp8_kv_fp8_%j.out
#SBATCH --error=slurm/logs/70b_fp8_kv_fp8_%j.err

# Verify the full fp8-weight + fp8-KV + TP=2 stack with a tiny prompt
# before re-launching the full exp1c_v2 grid.

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

echo "=== TP=2 fp8 weights + fp8 KV smoke on RedHatAI 70B ==="
uv run torchrun --standalone --nproc_per_node=2 scripts/smoke_fp8_load.py \
    --model RedHatAI/Meta-Llama-3-70B-Instruct-FP8 \
    --dtype bf16 \
    --kv-dtype fp8_e4m3 \
    --beam-width 4 \
    --max-new 8

echo
echo "=== smoke complete ==="

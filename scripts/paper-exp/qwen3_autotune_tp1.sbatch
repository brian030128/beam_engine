#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=qwen3_autotune_tp1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/qwen3_autotune_tp1_%j.out
#SBATCH --error=slurm/logs/qwen3_autotune_tp1_%j.err

# Auto-tune the bs_kernel cost-model overhead coefficients for
# Qwen/Qwen3-4B at TP=1 on H100. Writes to the model-keyed cache
# `coeffs-<gpu>-Qwen_Qwen3-4B.json` (see calibrate._cache_path).
# Existing Llama-tuned legacy cache `coeffs-<gpu>.json` is untouched.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf

mkdir -p slurm/logs

echo "=== autotune Qwen3-4B at TP=1 started $(date -Iseconds) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 1"

uv run python -m beam_engine.methods.bs_kernel.autotune \
    --model Qwen/Qwen3-4B \
    --save

echo "=== exit rc=$? ==="
echo "=== cache files ==="
ls -la ~/.cache/beam_engine/ 2>/dev/null

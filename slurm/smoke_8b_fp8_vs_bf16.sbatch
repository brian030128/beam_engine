#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=8b_fp8_vs_bf16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/8b_fp8_vs_bf16_%j.out
#SBATCH --error=slurm/logs/8b_fp8_vs_bf16_%j.err

# Sanity-check: same prompts on the 8B FP8 checkpoint vs the bf16
# baseline, both on a single H100, both through the paged backend.
# Token-level continuations for greedy decode should be (close to)
# identical for the first handful of tokens. Diverging early would
# point at a loader / quantization bug.
#
# Model names live only here, as CLI args.

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

echo "=== 8B bf16 baseline (paged, bf16 KV) ==="
uv run python scripts/run_tp_beam_search.py \
    --model meta-llama/Llama-3.1-8B \
    --backend paged \
    --dtype bf16 \
    --beam-width 4 \
    --max-new 8 \
    --prompt "The capital of France is"

echo
echo "=== 8B fp8 (paged, fp8 KV) ==="
uv run python scripts/run_tp_beam_search.py \
    --model RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 \
    --backend paged \
    --dtype bf16 \
    --kv-dtype fp8_e4m3 \
    --beam-width 4 \
    --max-new 8 \
    --prompt "The capital of France is"

echo
echo "=== sanity smoke complete ==="

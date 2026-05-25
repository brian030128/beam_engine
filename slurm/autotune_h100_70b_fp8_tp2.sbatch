#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=at_70b_fp8_tp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/autotune_70b_fp8_tp2_%j.out
#SBATCH --error=slurm/logs/autotune_70b_fp8_tp2_%j.err

# Fit (share_extra_us, dual_pool_extra_us, dec_tail_extra_us,
# dec_tail_per_tail_page_us) on H100 TP=2 using
# RedHatAI/Meta-Llama-3-70B-Instruct-FP8 (bf16 activations).
#
# Why: the first 70B-fp8 exp2 ablation (job 206430) showed the picker
# choosing SHARED_2L_DEC_TAIL on a cell where SHARED_2L_1POOL is 4.4%
# faster. Coefficients were last fit on bf16 1B/8B — the fp8 path has
# different per-tile compute & HBM ratios (tensor cores saturate at
# smaller M/N; weight HBM bandwidth is halved). Without recalibration
# every 70B-fp8 paper-exp number will under-report by 4-5% on cells
# where DEC_TAIL is mis-picked.
#
# The autotune cache is keyed by (gpu, model, tp_size), so this writes
# alongside (does not overwrite) the existing 1B / 8B caches.

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

MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"

echo "=== autotune started $(date -Iseconds) ==="
echo "model: $MODEL"
echo "GPUs:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
echo "dtype: $BE_DTYPE"
echo

uv run torchrun --standalone --nproc_per_node=2 \
    -m beam_engine.methods.bs_kernel.autotune \
    --model "$MODEL" \
    --dtype bf16 \
    --max-new 16 \
    --save

echo
echo "=== autotune cache after fit ==="
ls -la ~/.cache/beam_engine/ 2>/dev/null | grep -i 70b || true

echo
echo "Done."

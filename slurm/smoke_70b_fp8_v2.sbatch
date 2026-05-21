#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=70b_fp8_smoke_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/70b_fp8_smoke_v2_%j.out
#SBATCH --error=slurm/logs/70b_fp8_smoke_v2_%j.err

# Minimal smoke: load RedHatAI/Meta-Llama-3-70B-Instruct-FP8 across 2x H100
# (TP=2), run beam_search once via paged and bs_kernel, verify both
# backends produce identical tokens. Validates the fp8 loader + fp8
# linear forward path end-to-end against the real checkpoint.

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# Ephemeral HF cache (70B weights too large for /work hard quota).
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== fp8 loader unit test (synthetic fixture) ==="
uv run python tests/test_fp8_load.py

echo
echo "=== TP=2 fp8 load + beam_search smoke on 70B ==="
uv run torchrun --standalone --nproc_per_node=2 scripts/smoke_fp8_load.py \
    --model RedHatAI/Meta-Llama-3-70B-Instruct-FP8 \
    --dtype bf16 \
    --beam-width 4 \
    --max-new 8

echo
echo "=== smoke complete ==="

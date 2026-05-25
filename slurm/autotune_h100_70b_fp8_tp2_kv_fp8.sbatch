#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=at_70b_fp8_kvfp8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/autotune_70b_fp8_tp2_kvfp8_%j.out
#SBATCH --error=slurm/logs/autotune_70b_fp8_tp2_kvfp8_%j.err

# Same as autotune_h100_70b_fp8_tp2.sbatch but with KV dtype = fp8.
# After the fp8-KV path landed (Triton append kernel + flashinfer
# uint8↔fp8 dispatch patch), we observed the picker stays on
# SHARED_2L_1POOL on 70B-fp8/multi_few_shot/B=8 even though forced
# 2L_DEC_TAIL is 28% faster. Re-running the autotune-as-regret-reporter
# under fp8 KV will reveal whether the regret on the calibration grid
# is small (then the miss is grid-coverage) or large (then the
# coefficients need fp8-specific calibration).

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
export BE_KV_DTYPE=fp8_e4m3

MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"

echo "=== autotune (fp8 KV) started $(date -Iseconds) ==="
echo "model: $MODEL  (dtype=bf16, kv_dtype=fp8_e4m3)"
echo

# Force a fresh calibration so the new per_tile L_kv probe runs and
# populates ``per_tile_per_kv_us``. Without this the cached coefficients
# (from before the L_kv-aware refactor) would default the slope to 0
# and the L_kv-aware cost expressions would behave as before.
rm -f ~/.cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3*.json
echo "[autotune] forced recalibration"

uv run torchrun --standalone --nproc_per_node=2 \
    -m beam_engine.methods.bs_kernel.autotune \
    --model "$MODEL" \
    --dtype bf16 \
    --kv-dtype fp8_e4m3 \
    --max-new 16

echo "=== done $(date -Iseconds) ==="

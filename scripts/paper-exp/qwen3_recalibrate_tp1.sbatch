#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=q3_recal_tp1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/q3_recal_tp1_%j.out
#SBATCH --error=slurm/logs/q3_recal_tp1_%j.err

# Full force-recalibrate + autotune for Qwen3-4B at TP=1 with the new
# physics-grounded probes:
#   - measure_decode_per_kv_us: small-BS pairs added (B*K=64,128) to fit
#     decode launch + per-kv slope at K=16 regime
#   - measure_merge_us: 2-point line fit (launch + bw_us_per_row)
#   - measure_bw_efficiency_floor: empirical replacement for the 0.5 floor
# Then re-autotune to fit residual slacks (share_extra_us, dual_pool_extra_us,
# dec_tail_extra_us) against the autotune grid.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf

mkdir -p slurm/logs

echo "=== q3 recal+autotune (TP=1) started $(date -Iseconds) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Force-recalibrate the H100 legacy GPU cache so the new probes fire.
# autotune below pulls from this when starting.
uv run python -m beam_engine.methods.bs_kernel.calibrate --force 2>&1 | tee -a /dev/null

echo
echo "=== autotuning Qwen3-4B at TP=1 ==="
uv run python -m beam_engine.methods.bs_kernel.autotune \
    --model Qwen/Qwen3-4B \
    --save

echo "=== exit rc=$? ==="
echo "=== cached files ==="
ls -la ~/.cache/beam_engine/ 2>/dev/null

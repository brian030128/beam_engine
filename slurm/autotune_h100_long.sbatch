#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=at_h100_lng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/autotune_h100_long_%j.out
#SBATCH --error=slurm/logs/autotune_h100_long_%j.err

# Re-fit cost-model coefficients on Llama-3.2-1B with max-new=1024.
# Motivation: prior fit used max-new=256, suffix_len=128. The picker
# evaluates strategies at suffix_len = max_new/2; at gov_report
# max_new=2048 the runtime suffix_len is ~1024, but the fitted
# coefficients only saw signal at suffix_len=128 — so dec_tail_extra_us
# is under-fit for the long-decode regime. AUTOTUNE_GRID also gains the
# K=32/B=16 cell to give the fitter signal at the exact mispick point.

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CACHE=~/.cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json
TS=$(date +%Y%m%d-%H%M%S)
BACKUP="${CACHE}.bak.before_long_${TS}"

echo "=== backing up cache → $BACKUP ==="
cp "$CACHE" "$BACKUP"
echo
echo "=== current cache (will be overwritten) ==="
cat "$CACHE"
echo

uv run python -m beam_engine.methods.bs_kernel.autotune \
    --model meta-llama/Llama-3.2-1B \
    --max-new 1024 \
    --save

echo
echo "=== updated cache ==="
cat "$CACHE"
echo
echo "Done. To revert: cp $BACKUP $CACHE"

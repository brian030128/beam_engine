#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=at_h100_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/autotune_h100_8b_%j.out
#SBATCH --error=slurm/logs/autotune_h100_8b_%j.err

# Re-fit (share_extra_us, dual_pool_extra_us, dec_tail_extra_us) on H100
# using Llama-3.1-8B as the measurement model.
#
# Motivation: the phase-paper sweep + forced-strategy ablation on 8B
# K=16 (multi_few_shot/chain_reasoning/document) showed the picker
# chooses DEC_TAIL where SHARED_2L_1POOL is 17-20% faster. The
# coefficient cache was last fit on Llama-3.2-1B, where DEC_TAIL's
# per-launch overhead is amortised across only 16 layers; on 8B's 32
# layers it doubles. So the cached `dec_tail_extra_us=5.0 µs` is too
# small to penalise DEC_TAIL on 8B, and the picker mis-picks.
#
# Coefficients are GPU-keyed (not model-keyed), so this overwrites the
# values previously fit for 1B. We back up the cache first; if 1B
# regresses on the phase_paper cells, restore the .bak.before_8b file.

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CACHE=~/.cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json
BACKUP="${CACHE}.bak.before_8b_$(date +%Y%m%d-%H%M%S)"

echo "=== backing up cache → $BACKUP ==="
cp "$CACHE" "$BACKUP"

echo
echo "=== current cache (will be overwritten) ==="
cat "$CACHE"
echo

uv run python -m beam_engine.methods.bs_kernel.autotune \
    --model meta-llama/Llama-3.1-8B \
    --max-new 256 \
    --save

echo
echo "=== updated cache ==="
cat "$CACHE"

echo
echo "=== diff against backup ==="
diff "$BACKUP" "$CACHE" || true

echo
echo "Done. To revert: cp $BACKUP $CACHE"

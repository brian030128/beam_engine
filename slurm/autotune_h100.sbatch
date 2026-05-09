#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=autotune_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:35:00
#SBATCH --output=slurm/logs/autotune_h100_%j.out
#SBATCH --error=slurm/logs/autotune_h100_%j.err

# Re-fit (share_extra_us, dual_pool_extra_us) on H100 with the
# expanded autotune grid (now includes B=8 and B=32 cells, plus 3L
# probe modes and a wider DUAL_POOL_EXTRA_GRID_US).
#
# Replaces the prior cache (B=1-only fit, dual_pool_extra_us=0) which
# made the picker choose 3L_2POOL at K=64/B=32 even though 3L_1POOL is
# 25-30% faster on this device.

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== current cache (will be overwritten) ==="
cat ~/.cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json
echo

# Use Llama-3.2-1B (matches diagnose_cell.py and bench_full_cell — the
# same model that exhibits the mispick at B=32). max-new=64 so 3L
# strategies fire on enough decode steps to be measured.
uv run python -m beam_engine.methods.bs_kernel.autotune \
    --model meta-llama/Llama-3.2-1B \
    --max-new 64 \
    --save

echo
echo "=== updated cache ==="
cat ~/.cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json

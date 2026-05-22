#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=recalib_cta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/recalib_cta_%j.out
#SBATCH --error=slurm/logs/recalib_cta_%j.err

# Recalibrate cost-model coefficients on H100 for the CTA-granularity
# wave model on branch cost-model/cta-granularity. The added probe is
# measure_per_tile_per_kv_us, which fits per_tile_per_kv_us[T] from two
# L_kv points. The CTA-granularity formula consumes this slope as
# τ_elem = slope · cta_tile_kv.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

mkdir -p slurm/logs

echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo
echo "=== old cache (will be backed up + overwritten) ==="
CACHE=~/.cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json
if [ -f "$CACHE" ]; then
    cat "$CACHE"
    cp "$CACHE" "$CACHE.bak.$(date +%Y%m%d-%H%M%S)"
fi
echo

uv run python -c "
import torch
from beam_engine.methods.bs_kernel.calibrate import calibrate, _print_coeffs
device = torch.device('cuda')
coeffs = calibrate(device, force=True, verbose=True)
print()
_print_coeffs(coeffs)
"

echo
echo "=== new cache ==="
cat "$CACHE"

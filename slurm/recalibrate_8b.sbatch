#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=recal_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/recal_8b_%j.out
#SBATCH --error=slurm/logs/recal_8b_%j.err

# Re-run all physical probes in calibrate.py to refresh the H100 coefficient
# cache after the cost-model formula change (elemental sub-tile counting +
# extra decode_launch term). The probed constants (B_hbm, slope, decode
# per-(beam, KV-token), decode_launch_us, merge_launch_us) are device-level
# physics, unchanged by the formula refactor — but we re-measure to make
# sure the cache wasn't drifting.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

CACHE=~/.cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json
STAMP=$(date +%Y%m%d-%H%M%S)
BACKUP="${CACHE}.bak.before_paper_formula_${STAMP}"

echo "=== old cache (will be backed up to $BACKUP) ==="
[[ -f "$CACHE" ]] && cp "$CACHE" "$BACKUP" && cat "$CACHE"
echo

uv run python -m beam_engine.methods.bs_kernel.calibrate --force

echo
echo "=== new cache ==="
cat "$CACHE"

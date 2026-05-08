#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=profile_phases
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/profile_phases_%j.out
#SBATCH --error=slurm/logs/profile_phases_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# K=64 × L_p=8192 × B=32 × max_new=128 (half of the bench so wall stays
# under 5 min × 5 modes). Phase breakdown on the picker + 4 representative
# forced modes.
uv run python benchmarks/bs_kernel/profile_phases.py \
    --K 64 \
    --L_p 8192 \
    --B 32 \
    --max_new 128 \
    --modes picker SHARED_2L_1POOL SHARED_3L_1POOL SHARED_2L_DEC_TAIL PER_BEAM

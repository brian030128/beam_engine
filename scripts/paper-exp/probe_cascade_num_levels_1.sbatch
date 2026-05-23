#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=probe_L1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/probe_L1_%j.out
#SBATCH --error=slurm/logs/probe_L1_%j.err

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Force flashinfer to recompile the patched fused_cascade module.
# (JIT cache is keyed by source-file hashes so it will rebuild automatically.)
uv run python scripts/paper-exp/probe_cascade_num_levels_1.py

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=dump_modes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/dump_picks_modes_%j.out
#SBATCH --error=slurm/logs/dump_picks_modes_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Same grid + max_new as bench_modes so the pick distribution reflects
# the long-decode regime (per-beam tail grows from 0 → ~512 tokens).
uv run python benchmarks/bs_kernel/dump_picks_batched.py \
    --K 16 64 --L_p 8192 --B 1 16 32 --max_new 512

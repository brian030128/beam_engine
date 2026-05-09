#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=dump_batched
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/dump_picks_batched_%j.out
#SBATCH --error=slurm/logs/dump_picks_batched_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

uv run python benchmarks/bs_kernel/dump_picks_batched.py \
    --K 16 32 64 --L_p 8192 --B 1 2 4 8 --max_new 16

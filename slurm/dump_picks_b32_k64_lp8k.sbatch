#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=picks_b32k64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/picks_b32k64_%j.out
#SBATCH --error=slurm/logs/picks_b32k64_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

echo "=== nvidia-smi ==="
nvidia-smi
echo
echo "=== Running dump_picks_batched on B=32, K=64, L_p=8192 ==="
uv run python benchmarks/bs_kernel/dump_picks_batched.py \
    --K 64 --L_p 8192 --B 32 --max_new 16

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=ft_radix_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/test_radix_fastpath_%j.out
#SBATCH --error=slurm/logs/test_radix_fastpath_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

uv run python tests/test_radix_fastpath.py

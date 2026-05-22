#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=probe_cascade_thin_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/probe_cascade_thin_q_%j.out
#SBATCH --error=slurm/logs/probe_cascade_thin_q_%j.err

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

mkdir -p slurm/logs

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo

uv run python benchmarks/bs_kernel/probe_cascade_thin_q.py --sweep-l0-chunks

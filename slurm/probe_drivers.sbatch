#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal2
#SBATCH --job-name=drv_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:H200:1
#SBATCH --time=00:02:00
#SBATCH --output=slurm/logs/drv_probe_%j.out
#SBATCH --error=slurm/logs/drv_probe_%j.err

set -euo pipefail
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvidia-smi | head -3

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_var_splitk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/var_splitk_%j.out
#SBATCH --error=slurm/logs/var_splitk_%j.err
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export HF_HOME=/work/u4320956/hf
uv run python scripts/paper-exp/validate_variable_splitk.py

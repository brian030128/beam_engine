#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_varlen_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/varlen_merge_%j.out
#SBATCH --error=slurm/logs/varlen_merge_%j.err

# Validate the newly-bound variable_length_merge_states kernel (JIT compile
# + numerical correctness vs a float64 reference). No model load.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export HF_HOME=/work/u4320956/hf
uv run python scripts/paper-exp/validate_varlen_merge.py

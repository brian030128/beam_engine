#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=prof_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/profile_fasttree_vs_bs_8b_%j.out
#SBATCH --error=slurm/logs/profile_fasttree_vs_bs_8b_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export BE_MODEL=meta-llama/Llama-3.1-8B
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 8B model is bigger; B=8 at K=64/L_p=8192 keeps total VRAM ~40 GB
# (24 GB KV + 16 GB weights), fits comfortably on a single 80 GB H100.
uv run python benchmarks/bs_kernel/profile_fasttree_vs_bs.py \
    --K 64 --L_p 8192 --B 8 --max_new 256

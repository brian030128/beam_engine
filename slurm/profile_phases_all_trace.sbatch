#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=phases_trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/profile_phases_all_trace_%j.out
#SBATCH --error=slurm/logs/profile_phases_all_trace_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1

# Only bs_kernel — same cell as job 196321 — with sub-phase plan trace
# dumped at end of beam_search.
uv run python benchmarks/bs_kernel/profile_phases_all.py \
    --K 64 \
    --L_p 30000 \
    --B 16 \
    --max_new 256 \
    --methods bs_kernel

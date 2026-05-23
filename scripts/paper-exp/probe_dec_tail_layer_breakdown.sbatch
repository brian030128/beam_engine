#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=probe_dt_lay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/probe_dt_lay_%j.out
#SBATCH --error=slurm/logs/probe_dt_lay_%j.err

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_DEC_TAIL_KERNELS=1
# 8B's HF cache is at ~/.cache/huggingface (not /work/u4320956/hf which
# only contains the metadata stub). Leave HF_HOME unset to use default.

uv run python scripts/paper-exp/probe_dec_tail_layer_breakdown.py

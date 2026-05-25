#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_paged_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/paged_rerun_1b_%j.out
#SBATCH --error=slurm/logs/paged_rerun_1b_%j.err
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh; module load cuda/12.4; source .venv/bin/activate
export HF_HOME=/work/u4320956/hf
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="meta-llama/Llama-3.2-1B-Instruct"; export BE_DTYPE=bf16; export BE_KV_DTYPE=bf16
MODEL_TAG="Llama-3.2-1B-bf16"; NPROC=1; F2_B=16; F3_B=16
source scripts/paper-exp/paged_rerun_lib.sh

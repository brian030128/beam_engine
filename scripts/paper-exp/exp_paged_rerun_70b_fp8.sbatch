#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_paged_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/paged_rerun_70b_%j.out
#SBATCH --error=slurm/logs/paged_rerun_70b_%j.err
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh; module load cuda/12.4; source .venv/bin/activate
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"; mkdir -p "$HF_HOME"; trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_WORKSPACE_MB=1024
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"; export BE_DTYPE=bf16; export BE_KV_DTYPE=fp8_e4m3
MODEL_TAG="Llama-3-70B-Instruct-FP8"; NPROC=2; F2_B=4; F3_B=4
source scripts/paper-exp/paged_rerun_lib.sh

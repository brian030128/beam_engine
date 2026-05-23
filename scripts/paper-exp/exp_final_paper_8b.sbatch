#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_final_paper_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=05:00:00
#SBATCH --output=slurm/logs/final_paper_8b_%j.out
#SBATCH --error=slurm/logs/final_paper_8b_%j.err

# Final paper E2E table — Llama-3.1-8B (bf16 weights + bf16 KV, TP=1).
# See final_paper_lib.sh for the cell definitions + max-fit ladders.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME=/work/u4320956/hf
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16

MODEL_TAG="Llama-3.1-8B-bf16"
NPROC=1
SGLANG_CORE="paged fasttree deft mlca bs_kernel bsk_2l_1p bsk_2l_dt"
SGLANG_EXTRA=""
BATCHED_CORE="paged fasttree deft mlca bs_kernel bs_kernel_2l1p bs_kernel_2ldt"
BATCHED_EXTRA=""
source scripts/paper-exp/final_paper_lib.sh

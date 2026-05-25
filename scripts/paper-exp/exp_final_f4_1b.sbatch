#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f4_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=05:00:00
#SBATCH --output=slurm/logs/f4_1b_%j.out
#SBATCH --error=slurm/logs/f4_1b_%j.err

# Final-paper F4 (self_consistency) cell — Llama-3.2-1B (bf16, TP=1).
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
export BE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16

# Data draws for F4. Long-decode (mn=12k) timing is content-independent, so
# draws serve the example-count statistic + a small variance estimate, not the
# median. deft on F4 is very slow (~8 min/run at 1B), so keep this modest.
export F4_SEEDS="${F4_SEEDS:-0,1,2}"

MODEL_TAG="Llama-3.2-1B-bf16"
NPROC=1
SC_CORE="paged fasttree deft mlca bs_kernel bsk_2l_1p_single bsk_2l_dt"
SC_EXTRA=""
source scripts/paper-exp/final_f4_lib.sh

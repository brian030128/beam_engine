#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_final_paper_70b_bf16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4
#SBATCH --time=20:00:00
#SBATCH --output=slurm/logs/final_paper_70b_bf16_%j.out
#SBATCH --error=slurm/logs/final_paper_70b_bf16_%j.err

# Final-paper F1-F3 — Llama-3-70B-Instruct (bf16 weights + bf16 KV, TP=4),
# unquantized. Same scenarios/method-set/ladders as the 1B/8B runs (all
# methods work with bf16 KV, so fasttree/deft/mlca are in the gated set, no
# fp8 best-effort split). F4 self_consistency is intentionally skipped here.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# bf16 70B is ~140 GiB on disk → ephemeral cache, cleaned by the lib (EPHEMERAL).
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
export HF_HOME_EPHEMERAL=1
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_WORKSPACE_MB=1024
export BE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16

# Data draws: 4 resamples per cell (fewer than 1B/8B — 70B is the expensive
# model and timing is the most forward-bound here).
export SEEDS="${SEEDS:-0,1,2,3}"

MODEL_TAG="Llama-3-70B-bf16"
NPROC=4
SGLANG_CORE="paged fasttree deft mlca bs_kernel bsk_2l_1p bsk_2l_dt"
SGLANG_EXTRA=""
BATCHED_CORE="paged fasttree deft mlca bs_kernel bs_kernel_2l1p bs_kernel_2ldt"
BATCHED_EXTRA=""
source scripts/paper-exp/final_paper_lib.sh

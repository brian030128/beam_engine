#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_final_paper_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/final_paper_70b_%j.out
#SBATCH --error=slurm/logs/final_paper_70b_%j.err

# Final paper E2E table — Llama-3-70B-Instruct-FP8 (fp8 weights +
# fp8_e4m3 KV, TP=2). fasttree / deft crash on fp8 KV at 70B (project
# memory) so they are dropped. See final_paper_lib.sh for the max-fit
# ladders; the big cells shrink B further than the 1B/8B runs.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# Ephemeral cache for the 70B fp8 checkpoint (~70 GiB on disk).
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
export HF_HOME_EPHEMERAL=1   # final_paper_lib.sh removes $HF_HOME on exit
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3

MODEL_TAG="Llama-3-70B-Instruct-FP8"
NPROC=2
# CORE = the set known to work on fp8 KV (gates the B ladder). mlca /
# fasttree / deft are best-effort EXTRA: they have crashed on fp8 KV at
# 70B before, but the fp8-append path was fixed 2026-05-21 so we retry
# them in an isolated pass that can't poison the core results.
SGLANG_CORE="paged bs_kernel bsk_2l_1p bsk_2l_dt"
SGLANG_EXTRA="mlca fasttree deft"
BATCHED_CORE="paged bs_kernel bs_kernel_2l1p bs_kernel_2ldt"
BATCHED_EXTRA="mlca fasttree deft"
source scripts/paper-exp/final_paper_lib.sh

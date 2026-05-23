#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f4_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/f4_70b_%j.out
#SBATCH --error=slurm/logs/f4_70b_%j.err

# Final-paper F4 (self_consistency) cell — Llama-3-70B-Instruct-FP8
# (fp8 weights + fp8_e4m3 KV, TP=2). B=1 K=16 L_p=16K fits easily.
# mlca/fasttree/deft are best-effort EXTRA (isolated; fp8 KV crash-prone).
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
# 70B (64 qo heads) overflows the default 128 MiB FlashInfer scratch on the
# single-launch 16K prefill — give it more headroom (1 GiB covers K up to 64).
# Overridable so a submit can bump it further.
export BE_WORKSPACE_MB="${BE_WORKSPACE_MB:-1024}"
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3

MODEL_TAG="Llama-3-70B-Instruct-FP8"
NPROC=2
# Same F4 setting as 1B/8B for cross-model consistency: K=16, L_p=16K,
# B=1 (the lib default). NOTE: 70B is forward-bound at K=16, so paged
# edges 1p by ~1% here; the 1p-over-paged crossover lands at K=64 — kept
# as supplementary sweep data (F4_self_consistency_K{32,64}_...csv), not
# the headline cell. Override K with F4_K=... if needed.
SC_CORE="paged bs_kernel bsk_2l_1p_single bsk_2l_dt"
SC_EXTRA="mlca fasttree deft"
source scripts/paper-exp/final_f4_lib.sh

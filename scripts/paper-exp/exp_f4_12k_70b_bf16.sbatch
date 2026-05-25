#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f4_12k_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/f4_12k_70b_%j.out
#SBATCH --error=slurm/logs/f4_12k_70b_%j.err

# Final-paper F4 (self_consistency) cell — Llama-3-70B-Instruct
# (bf16 weights + bf16 KV, TP=4) — the paper-result 70B setting — with a
# LONG decode: max_new=12000, L_p=64K (paper per-model L_p). Trimmed method
# set (picker + paged + two forced bs_kernel strategies; no fasttree/deft/mlca).
# Trace OFF for fair timing; per-step strategy via BE_DUMP_STEP_TIMINGS.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"; trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_WORKSPACE_MB=2048   # 64K single-launch prefill scratch
export BE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16

export F4_MN=12000
export F4_REPEAT=2
export F4_LP=65536            # paper 70B F4 prefix length

MODEL_TAG="Llama-3-70B-bf16"
NPROC=4
export BE_DUMP_STEP_TIMINGS="benchmarks/bs_kernel/results/paper-exp/final_paper/${MODEL_TAG}/step_timings_mn12000"
SC_CORE="paged bs_kernel bsk_2l_1p_single bsk_2l_dt"
SC_EXTRA=""
source scripts/paper-exp/final_f4_lib.sh

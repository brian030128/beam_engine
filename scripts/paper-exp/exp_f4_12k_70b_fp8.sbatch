#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f4_12k_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/f4_12k_70b_%j.out
#SBATCH --error=slurm/logs/f4_12k_70b_%j.err

# Final-paper F4 (self_consistency) cell — Llama-3-70B-Instruct-FP8
# (fp8 weights + fp8_e4m3 KV, TP=2) — LONG decode: max_new=12000. Trimmed
# method set (picker + paged + two forced bs_kernel strategies; no
# fasttree/deft/mlca). Trace OFF for fair timing; per-step strategy via
# BE_DUMP_STEP_TIMINGS.
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
# 70B (64 qo heads) overflows the default 128 MiB FlashInfer scratch on the
# single-launch 16K prefill — give it 1 GiB. Overridable.
export BE_WORKSPACE_MB="${BE_WORKSPACE_MB:-1024}"
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3

export F4_MN=12000
export F4_REPEAT=2

MODEL_TAG="Llama-3-70B-Instruct-FP8"
NPROC=2
export BE_DUMP_STEP_TIMINGS="benchmarks/bs_kernel/results/paper-exp/final_paper/${MODEL_TAG}/step_timings_mn12000"
SC_CORE="paged bs_kernel bsk_2l_1p_single bsk_2l_dt"
SC_EXTRA=""
source scripts/paper-exp/final_f4_lib.sh

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f4_70b_bf16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4
#SBATCH --time=14:00:00
#SBATCH --output=slurm/logs/f4_70b_bf16_%j.out
#SBATCH --error=slurm/logs/f4_70b_bf16_%j.err

# F4 self_consistency for 70B bf16 (TP=4), L_p=64K K=16 B=1. All methods
# work at bf16 KV (full set in SC_CORE, no fp8 best-effort split).
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
export BS_KERNEL_TRACE_PLAN=1
export BE_WORKSPACE_MB=2048   # 64K single-launch prefill scratch
export BE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16

# Data draws for F4 at 70B. Long-decode timing is content-independent, so 2
# draws is enough for the example-count statistic. WARNING: deft on 70B F4 is
# ~83 min/RUN (5e6 ms); even D=2 makes deft dominate the walltime. If tight,
# either drop deft from SC_CORE here or set F4_SEEDS=0 (deft adds no info
# beyond the single catastrophic-baseline number we already report).
export F4_SEEDS="${F4_SEEDS:-0,1}"

MODEL_TAG="Llama-3-70B-bf16"
NPROC=4
export F4_LP=65536
SC_CORE="paged fasttree deft mlca bs_kernel bsk_2l_1p_single bsk_2l_dt"
SC_EXTRA=""
source scripts/paper-exp/final_f4_lib.sh

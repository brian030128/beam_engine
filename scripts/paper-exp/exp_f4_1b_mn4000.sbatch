#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f4_1b_mn4k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:30:00
#SBATCH --output=slurm/logs/f4_1b_mn4000_%j.out
#SBATCH --error=slurm/logs/f4_1b_mn4000_%j.err

# F4 self_consistency on Llama-3.2-1B (bf16, TP=1) with a LONG decode:
# max_new=4000 (vs 256 in the paper cell). Same static-2L best-of-N shape
# (K=16 independent sampled tails over a shared L_p=32K prefix, B=1), so
# the picker should stay on SHARED_2L_1POOL (single-launch) the whole way.
# Writes F4_self_consistency_K16_B1_Lp32768_mn4000.csv alongside the others.
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
export BE_DTYPE=bf16            # 1B head_dim=64; FA3/SM90 build bug → force FA2
export BE_KV_DTYPE=bf16
export BE_FORCE_FA2=1

# F4 cell params (read by final_f4_lib.sh): long-decode override.
export F4_K=16
export F4_B=1
export F4_LP=32768
export F4_MN=4000

MODEL_TAG="Llama-3.2-1B-bf16"
NPROC=1
SC_CORE="paged fasttree deft mlca bs_kernel bsk_2l_1p_single bsk_2l_dt"
SC_EXTRA=""
source scripts/paper-exp/final_f4_lib.sh

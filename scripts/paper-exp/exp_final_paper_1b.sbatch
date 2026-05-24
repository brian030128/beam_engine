#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_final_paper_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/final_paper_1b_%j.out
#SBATCH --error=slurm/logs/final_paper_1b_%j.err

# Final paper E2E table — Llama-3.2-1B (bf16 weights + bf16 KV, TP=1).
# Three cells, matching the paper's evaluation scenarios:
#   F1 multi_doc_qa     K=128 B=1   mn=256  (long shared prefix ~85K)
#   F2 multi_few_shot   K=128 B=<=64 mn=256 (short prefix ~4K, large B)
#   F3 beam_search      K=64  B=<=16 L_p=40000 mn=256 (multi-level beam)
# Max-fit-per-model: F2/F3 walk a B ladder (largest first) and accept the
# first config where ALL methods fit. Actual (K,B) is in the CSV name.

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
# 1B is head_dim=64: the flashinfer FA3/SM90 batch_prefill kernel fails to
# compile for hd=64 ("No eligible GMMA operator"), breaking paged/fasttree/
# tree/mlca on a cold build. Force FA2 prefill (compiles+runs for all hd on
# Hopper). 8B/70B (hd=128) keep FA3 — do NOT set this there.
export BE_FORCE_FA2=1

MODEL_TAG="Llama-3.2-1B-bf16"
NPROC=1
SGLANG_CORE="paged fasttree deft mlca bs_kernel bsk_2l_1p bsk_2l_dt"
SGLANG_EXTRA=""
BATCHED_CORE="paged fasttree deft mlca bs_kernel bs_kernel_2l1p bs_kernel_2ldt"
BATCHED_EXTRA=""
source scripts/paper-exp/final_paper_lib.sh

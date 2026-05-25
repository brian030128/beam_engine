#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_mdqa_b2_40k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/mdqa_b2_40k_%j.out
#SBATCH --error=slurm/logs/mdqa_b2_40k_%j.err

# multi_doc_qa at K=128 B=4 on Llama-3-70B-FP8 (512 leaves, 4 doc-sets).
# Uses the fp8-fixed mlca + the FlashInfer-style L0 split-K. All methods
# that support fp8 KV (paged, mlca, bs_kernel + forced 1p/dt).
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
export BE_WORKSPACE_MB=1024
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3
export BE_HOTPOTQA_MULTI_Q_PATH=data/hotpotqa_multi_q_K128_40k.jsonl

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/final_paper/Llama-3-70B-Instruct-FP8"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run_mdqa_b2_40k_costfix.log"
csv="$OUT_DIR/F1e_multi_doc_qa_K128_B2_Lp40k_costfix_mn256.csv"
echo "=== multi_doc_qa K=128 B=2 L_p=40K 70B-fp8 $(date -Iseconds) ===" | tee -a "$LOG"

torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods bs_kernel bsk_2l_dt mlca \
    --no-stage2 --max-new 256 --repeat 3 --warmup \
    --scenarios multi_doc_qa --k-override 128 --b-override 2 \
    --out "$csv" 2>&1 | tee -a "$LOG"

echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"

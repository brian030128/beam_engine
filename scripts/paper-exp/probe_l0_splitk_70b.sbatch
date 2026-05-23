#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_splitk_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/splitk_70b_%j.out
#SBATCH --error=slurm/logs/splitk_70b_%j.err

# Verify the FlashInfer-style L0 split-K change (cascade.py) closes the
# 70B F1 multi_doc_qa gap to mlca. Compares bs_kernel (picker->DEC_TAIL)
# and forced bsk_2l_dt against mlca on the 80K-prefix/K=128 cell.
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
export BE_WORKSPACE_MB=2048
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3
export BE_HOTPOTQA_MULTI_Q_PATH=data/hotpotqa_multi_q_K128_80k.jsonl

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/probe_splitk/Llama-3-70B-Instruct-FP8"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"
echo "=== L0 split-K probe (FlashInfer-style) $(date -Iseconds) ===" | tee -a "$LOG"

torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods mlca bs_kernel bsk_2l_dt \
    --no-stage2 --max-new 256 --repeat 3 --warmup \
    --scenarios multi_doc_qa --k-override 128 --b-override 1 \
    --out "$OUT_DIR/F1_splitk_K128_B1.csv" 2>&1 | tee -a "$LOG"

echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"

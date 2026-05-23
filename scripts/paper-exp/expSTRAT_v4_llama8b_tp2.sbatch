#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_expSTRATv4_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/paper_expSTRATv4_llama8b_tp2_%j.out
#SBATCH --error=slurm/logs/paper_expSTRATv4_llama8b_tp2_%j.err

# expSTRAT v4 — fp8 KV cache enables bigger R / longer tail cells on
# Llama-3.1-8B. Counterpart to the Qwen3-4B v4 sbatch.
#
#   tag | scenario              | K   | B  | max_new | R    | mid-tail
#   BR5 | multi_chain_reasoning | 128 | 32 |     256 | 4096 | ≈128
#   BR6 | multi_chain_reasoning | 128 | 64 |     128 | 8192 | ≈64
#   BR8 | multi_chain_reasoning | 128 | 16 |     512 | 2048 | ≈256

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="meta-llama/Llama-3.1-8B"
export BE_KV_DTYPE=fp8_e4m3

MODEL_TAG="Llama-3.1-8B"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/expSTRATv4/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== expSTRATv4 ${MODEL_TAG} TP=2 started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, kv=$BE_KV_DTYPE)"
    echo
} | tee "$LOG"

METHODS="paged bs_kernel bsk_2l_1p bsk_2l_dt"

run_cell () {
    local tag="$1"; shift
    local scenario="$1"; shift
    local K="$1"; shift
    local B="$1"; shift
    local mn="$1"; shift
    local csv="$OUT_DIR/${scenario}_K${K}_B${B}_mn${mn}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== ${tag}: ${scenario}  (K=${K}, B=${B}, max_new=${mn}, R=$((B*K)), tp=2, kv=fp8) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS \
        --paper-exact --no-stage2 \
        --max-new "$mn" --repeat 1 --warmup \
        --scenarios "$scenario" --k-override "$K" --b-override "$B" \
        --out "$csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (cell $tag failed; continuing)" | tee -a "$LOG"
}

run_cell BR5 multi_chain_reasoning 128 32 256
run_cell BR6 multi_chain_reasoning 128 64 128
run_cell BR8 multi_chain_reasoning 128 16 512

{
    echo
    echo "=== done $(date -Iseconds) ==="
} | tee -a "$LOG"

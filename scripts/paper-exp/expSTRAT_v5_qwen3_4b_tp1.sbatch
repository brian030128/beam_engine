#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_expSTRATv5_q3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/paper_expSTRATv5_qwen3_4b_tp1_%j.out
#SBATCH --error=slurm/logs/paper_expSTRATv5_qwen3_4b_tp1_%j.err

# expSTRAT v5 — single-GPU fp8 weights + fp8 KV on Qwen3-4B-FP8.
# Tests whether the picker / forced-strategy ratios shift when running
# on TP=1 (no NCCL overhead) with full fp8.
#
#   tag | scenario              | K   | B  | max_new | R    | mid-tail
#   BR1 | multi_chain_reasoning |  64 | 32 |     256 | 2048 | ≈128
#   BR3 | multi_chain_reasoning |  64 | 64 |     128 | 4096 | ≈64
#   BR4 | multi_chain_reasoning | 128 | 32 |     128 | 4096 | ≈64

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export HF_HOME=/work/u4320956/hf
export BE_MODEL="Qwen/Qwen3-4B-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3

MODEL_TAG="Qwen3-4B-FP8"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/expSTRATv5/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== expSTRATv5 ${MODEL_TAG} TP=1 started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=1, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 1"
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
        echo "=== ${tag}: ${scenario}  (K=${K}, B=${B}, max_new=${mn}, R=$((B*K)), tp=1, fp8) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=1 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS \
        --paper-exact --no-stage2 \
        --max-new "$mn" --repeat 1 --warmup \
        --scenarios "$scenario" --k-override "$K" --b-override "$B" \
        --out "$csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (cell $tag failed; continuing)" | tee -a "$LOG"
}

run_cell BR1 multi_chain_reasoning  64 32 256
run_cell BR3 multi_chain_reasoning  64 64 128
run_cell BR4 multi_chain_reasoning 128 32 128

{
    echo
    echo "=== done $(date -Iseconds) ==="
} | tee -a "$LOG"

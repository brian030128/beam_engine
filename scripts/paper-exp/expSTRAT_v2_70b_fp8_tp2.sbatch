#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_expSTRATv2_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/paper_expSTRATv2_70b_fp8_tp2_%j.out
#SBATCH --error=slurm/logs/paper_expSTRATv2_70b_fp8_tp2_%j.err

# expSTRAT v2 — bigger-R Family B retry on Llama-3-70B-FP8 with fp8 KV.
# Same cells as the smaller models; fp8 KV halves footprint and keeps
# R=2048-4096 cells inside 80 GiB/rank.
#
#   tag | scenario              | K  | B  | max_new | R=B·K
#   BR1 | multi_chain_reasoning | 64 | 32 |     256 |  2048
#   BR2 | multi_few_shot        | 64 | 32 |     256 |  2048
#   BR3 | multi_chain_reasoning | 64 | 64 |     128 |  4096

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3
export BS_KERNEL_TRACE_PLAN=1

MODEL_TAG="Llama-3-70B-FP8"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/expSTRATv2/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== expSTRATv2 ${MODEL_TAG} TP=2 started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
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
        echo "=== ${tag}: ${scenario}  (K=${K}, B=${B}, max_new=${mn}, R=$((B*K)), tp=2) ==="
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

run_cell BR1 multi_chain_reasoning 64 32 256
run_cell BR2 multi_few_shot        64 32 256
run_cell BR3 multi_chain_reasoning 64 64 128

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

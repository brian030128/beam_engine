#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_70b_b4_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=02:30:00
#SBATCH --output=slurm/logs/pe_70b_b4_batch_%j.out
#SBATCH --error=slurm/logs/pe_70b_b4_batch_%j.err

# Extend the multi_few_shot K=32 cell to larger batches on 70B fp8 TP=2.
# Current B4 = (K=32, B=4); sweep B ∈ {8, 16, 32}. Full method set so
# paged-vs-picker-vs-cascade comparison stays apples-to-apples with B4.

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
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3

MODEL_TAG="Llama-3-70B-Instruct-FP8"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/picker_demo_2l1p_single_vs_2ldt/${MODEL_TAG}_8b_b3b4b5"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run_batch_sweep.log"

{
    echo "=== 70B fp8 | multi_few_shot K=32 batch sweep | $(date -Iseconds) ==="
    echo "model:   $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo "cells:   (K=32, B=8) (K=32, B=16) (K=32, B=32)  scenario=multi_few_shot mn=256"
} | tee -a "$LOG"

echo "=== recalibrating coefficients ===" | tee -a "$LOG"
uv run python -m beam_engine.methods.bs_kernel.calibrate --force 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"

SGLANG_METHODS="paged fasttree deft bs_kernel bsk_2l_1p_single bsk_2l_dt"

run_sglang_cell () {
    local tag="$1" scenario="$2" K="$3" B="$4" mn="$5"
    local csv="$OUT_DIR/${tag}_${scenario}_K${K}_B${B}_mn${mn}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== ${tag}: ${scenario}  (K=${K}, B=${B}, mn=${mn}, R=$((B*K))) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    rm -f "$csv"
    local first=1
    for m in $SGLANG_METHODS; do
        local tmp="$OUT_DIR/${tag}_${m}.csv"
        echo "  --- method: $m ---" | tee -a "$LOG"
        torchrun --standalone --nproc_per_node=2 \
            benchmarks/bs_kernel/bench_sglang_e2e.py \
            --methods "$m" \
            --paper-exact --no-stage2 \
            --max-new "$mn" --repeat 3 --warmup \
            --scenarios "$scenario" --k-override "$K" --b-override "$B" \
            --out "$tmp" \
            2>&1 | tee -a "$LOG" \
            || { echo "  ($m failed; continuing)" | tee -a "$LOG"; continue; }
        if [[ $first -eq 1 ]]; then
            [[ -f "$tmp" ]] && cp "$tmp" "$csv" && first=0
        else
            [[ -f "$tmp" ]] && tail -n +2 "$tmp" >> "$csv"
        fi
    done
}

run_sglang_cell B4_B8  multi_few_shot 32  8 256
run_sglang_cell B4_B16 multi_few_shot 32 16 256
run_sglang_cell B4_B32 multi_few_shot 32 32 256

{
    echo
    echo "=== done $(date -Iseconds) ==="
} | tee -a "$LOG"

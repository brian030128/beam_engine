#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_exp1c_v2_q3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/paper_exp1c_v2_qwen3_%j.out
#SBATCH --error=slurm/logs/paper_exp1c_v2_qwen3_%j.err

# exp1c_v2 — picker vs forced (FUSED, DEC_TAIL) across all four
# scenarios on Qwen/Qwen3-4B with TP=2 (paper-grid configuration).
# Rerun after the calibration + cost-model update.
#
# Cells:
#   scenario              | (B,  K, max_new) | bench harness
#   ----------------------+------------------+---------------------
#   multi_level_system    | (4,  64,  256)   | bench_sglang_e2e.py
#   multi_chain_reasoning | (16, 64,  256)   | bench_sglang_e2e.py
#   multi_few_shot        | (16, 64,  256)   | bench_sglang_e2e.py
#   beam_search           | (8,  32, 2048)   | bench_batched.py   (gov_report)
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp1c_v2/Qwen3-4B/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export HF_HOME=/work/u4320956/hf
export BE_MODEL="Qwen/Qwen3-4B"

MODEL_TAG="Qwen3-4B"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1c_v2/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== exp1c_v2 ${MODEL_TAG} started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2)"
    echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
    echo "HF_HOME:  $HF_HOME"
    echo
} | tee "$LOG"

SGLANG_METHODS="bs_kernel bsk_2l_1p bsk_2l_dt"
BATCHED_METHODS="bs_kernel bs_kernel_2l1p bs_kernel_2ldt"

run_sglang_cell () {
    local scenario="$1"; shift
    local B="$1"; shift
    local csv="$OUT_DIR/${scenario}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== cell: ${scenario}  (B=${B}, K=64, max_new=256) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $SGLANG_METHODS \
        --paper-exact --no-stage2 \
        --max-new 256 --repeat 1 --warmup \
        --scenarios "$scenario" --k-override 64 --b-override "$B" \
        --out "$csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (cell $scenario B=$B failed; continuing)" | tee -a "$LOG"
}

run_beam_search_cell () {
    local K="$1"; shift
    local L_p="$1"; shift
    local B="$1"; shift
    local max_new="$1"; shift
    local csv="$OUT_DIR/beam_search.csv"
    {
        echo
        echo "=========================================================="
        echo "=== cell: beam_search  (K=${K}, L_p=${L_p}, B=${B}, max_new=${max_new}) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    local first=1
    for m in $BATCHED_METHODS; do
        echo "  --- method: $m ---" | tee -a "$LOG"
        local tmp="$OUT_DIR/beam_search_${m}.csv"
        torchrun --standalone --nproc_per_node=2 \
            benchmarks/bs_kernel/bench_batched.py \
            --K "$K" --L_p "$L_p" --B "$B" --max_new "$max_new" \
            --prompts-file data/gov_report.jsonl \
            --warmup \
            --methods "$m" \
            --out "$tmp" \
            2>&1 | tee -a "$LOG" \
            || { echo "  (beam_search $m failed; continuing)" | tee -a "$LOG"; continue; }
        if [[ $first -eq 1 ]]; then
            [[ -f "$tmp" ]] && cp "$tmp" "$csv" && first=0
        else
            [[ -f "$tmp" ]] && tail -n +2 "$tmp" >> "$csv"
        fi
    done
}

run_sglang_cell multi_level_system     4
run_sglang_cell multi_chain_reasoning 16
run_sglang_cell multi_few_shot        16
run_beam_search_cell 32 8192 8 2048

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

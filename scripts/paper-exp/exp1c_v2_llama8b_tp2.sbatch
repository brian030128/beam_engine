#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_exp1c_v2_8b_tp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/paper_exp1c_v2_8b_tp2_%j.out
#SBATCH --error=slurm/logs/paper_exp1c_v2_8b_tp2_%j.err

# exp1c_v2 (TP=2) — picker vs forced (FUSED, DEC_TAIL) across all four
# scenarios on Llama-3.1-8B with TP=2. Runs after the cascade-wrapper
# refactor for DEC_TAIL prefix and the new paper-spec cost model.
#
# Cells (per-prompt B; TP=2 splits the model across 2 GPUs):
#   scenario              | (B,  K, max_new) | bench harness
#   ----------------------+------------------+---------------------
#   multi_level_system    | (4,  64,  256)   | bench_sglang_e2e.py
#   multi_chain_reasoning | (8,  64,  256)   | bench_sglang_e2e.py
#   multi_few_shot        | (8,  64,  256)   | bench_sglang_e2e.py
#   beam_search           | (8,  32, 2048)   | bench_batched.py   (gov_report)

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="meta-llama/Llama-3.1-8B"

MODEL_TAG="Llama-3.1-8B"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1c_v2_tp2/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== exp1c_v2 ${MODEL_TAG} TP=2 started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2)"
    echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
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
        echo "=== cell: ${scenario}  (B=${B}, K=64, max_new=256, tp=2) ==="
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
        echo "=== cell: beam_search  (K=${K}, L_p=${L_p}, B=${B}, max_new=${max_new}, tp=2) ==="
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

run_sglang_cell multi_level_system    4
run_sglang_cell multi_chain_reasoning 8
run_sglang_cell multi_few_shot        8
run_beam_search_cell 32 8192 8 2048

{
    echo
    echo "=== done $(date -Iseconds) ==="
} | tee -a "$LOG"

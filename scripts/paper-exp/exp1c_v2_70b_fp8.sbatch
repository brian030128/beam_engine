#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_exp1c_v2_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/paper_exp1c_v2_70b_fp8_%j.out
#SBATCH --error=slurm/logs/paper_exp1c_v2_70b_fp8_%j.err

# exp1c_v2 — picker vs forced (FUSED, DEC_TAIL) across all four
# scenarios on RedHatAI/Meta-Llama-3-70B-Instruct-FP8 with TP=2.
# Rerun after the calibration + cost-model update.
#
# Cells (B halved to fit per-rank fp8-weights + bf16-KV on H100 TP=2):
#   scenario              | (B,  K, max_new) | bench harness
#   ----------------------+------------------+---------------------
#   multi_level_system    | (4,  64,  256)   | bench_sglang_e2e.py
#   multi_chain_reasoning | (4,  64,  256)   | bench_sglang_e2e.py
#   multi_few_shot        | (4,  64,  256)   | bench_sglang_e2e.py
#   beam_search           | (4,  32,  768)   | bench_batched.py   (gov_report)
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp1c_v2/Llama-3-70B-FP8/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# 70B FP8 weights too large for /work — use ephemeral SLURM_TMPDIR.
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3   # fp8 KV cache via tvm_ffi/torch-c-dlpack-ext.
export BS_KERNEL_TRACE_PLAN=1

MODEL_TAG="Llama-3-70B-FP8"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1c_v2/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== exp1c_v2 ${MODEL_TAG} started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
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

run_sglang_cell multi_level_system    4
run_sglang_cell multi_chain_reasoning 4
run_sglang_cell multi_few_shot        4
run_beam_search_cell 32 8192 4 768

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

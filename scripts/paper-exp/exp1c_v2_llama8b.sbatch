#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_exp1c_v2_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/paper_exp1c_v2_8b_%j.out
#SBATCH --error=slurm/logs/paper_exp1c_v2_8b_%j.err

# exp1c_v2 — picker vs forced (FUSED, DEC_TAIL) across all four
# scenarios (3 sglang scenarios + beam_search long-decode) on
# Llama-3.1-8B (single GPU). Rerun after the calibration + cost-model
# update; the previous exp1c numbers in docs/paper_story.md are stale.
#
# For each cell we record:
#   - bs_kernel        (picker, free choice)
#   - bsk_2l_1p        (forced SHARED_2L_1POOL, fused prefill only)
#   - bsk_2l_dt        (forced SHARED_2L_DEC_TAIL, prefill root + decode tail)
#   plus the picker's strategy histogram via BS_KERNEL_TRACE_PLAN=1.
#
# Cells (all K=64 for the three multi_* scenarios, beam_search at K=32):
#   scenario              | (B,  K, max_new) | bench harness
#   ----------------------+------------------+---------------------
#   multi_level_system    | (4,  64,  256)   | bench_sglang_e2e.py
#   multi_chain_reasoning | (8,  64,  256)   | bench_sglang_e2e.py
#   multi_few_shot        | (8,  64,  256)   | bench_sglang_e2e.py
#   beam_search           | (4,  32, 2048)   | bench_batched.py   (gov_report)
#                          B halved 8→4 because 8B's 32 layers × per-rank
#                          KV @ K=32/B=8/max_new=2048 exceeds the H100's
#                          80 GiB. max_pages auto-sized (no override).
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp1c_v2/Llama-3.1-8B/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="meta-llama/Llama-3.1-8B"

MODEL_TAG="Llama-3.1-8B"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1c_v2/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== exp1c_v2 ${MODEL_TAG} started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=1)"
    echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "coeffs cache:"
    ls -la ~/.cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json 2>/dev/null || echo "  (none)"
    echo
} | tee "$LOG"

SGLANG_METHODS="bs_kernel bsk_2l_1p bsk_2l_dt"
BATCHED_METHODS="bs_kernel bs_kernel_2l1p bs_kernel_2ldt"

run_sglang_cell () {
    # run_sglang_cell <scenario> <B>
    local scenario="$1"; shift
    local B="$1"; shift
    local csv="$OUT_DIR/${scenario}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== cell: ${scenario}  (B=${B}, K=64, max_new=256) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $SGLANG_METHODS \
        --paper-exact --no-stage2 \
        --max-new 256 --repeat 1 --warmup \
        --scenarios "$scenario" --k-override 64 --b-override "$B" \
        --out "$csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (cell $scenario B=$B failed; continuing)" | tee -a "$LOG"
}

run_beam_search_cell () {
    # run_beam_search_cell <K> <L_p> <B> <max_new>
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
    # Run each method separately so BS_KERNEL_TRACE_PLAN dumps fire
    # cleanly per method (bench_batched.py reuses the bs_kernel
    # backend instance across methods otherwise mixing the traces).
    local first=1
    for m in $BATCHED_METHODS; do
        echo "  --- method: $m ---" | tee -a "$LOG"
        local tmp="$OUT_DIR/beam_search_${m}.csv"
        uv run python benchmarks/bs_kernel/bench_batched.py \
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
run_beam_search_cell 32 8192 4 2048

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
    echo "Strategy mix per cell is in $LOG (search for 'strategy mix')."
} | tee -a "$LOG"

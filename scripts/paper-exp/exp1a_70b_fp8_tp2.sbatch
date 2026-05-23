#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp1a_70b_fp8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/paper_exp1a_70b_fp8_%j.out
#SBATCH --error=slurm/logs/paper_exp1a_70b_fp8_%j.err

# TP=2 end-to-end timing on RedHatAI/Meta-Llama-3-70B-Instruct-FP8.
# Sister of exp1a_end_to_end_timing.sbatch but on the FP8-quantized
# 70B checkpoint, single scenario per cell (B halved vs the 8B
# variant since per-rank KV slab dominates at TP=2).
#
# Compute dtype bf16 (the original training dtype for Llama-3-70B);
# KV stays bf16 — fp8 KV is deferred until FlashInfer's
# append_paged_kv_cache dispatches fp8/uint8.
#
# Cells (K=64 to match the 8B paper-grid):
#   multi_level_system    | (B=4,  K=64)    [smallest, fits fine]
#   multi_chain_reasoning | (B=4,  K=64)    [halved from 8B's B=8]
#   multi_few_shot        | (B=4,  K=64)    [halved from 8B's B=8]
#
# bench_sglang_e2e loads the model once and runs all methods through it
# (so we pay the ~3-min 70B download exactly once across all cells).
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp1a_70b_fp8_tp2/

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

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1a_70b_fp8_tp2"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp1a_70b_fp8.log"

{
    echo "=== exp1a TP=2 70B-FP8 end-to-end timing started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=bf16)"
    echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
    echo
} | tee "$LOG"

METHODS="fasttree bs_kernel paged mlca"

run_cell () {
    # run_cell <scenario> <B> <csv>
    local scenario="$1"; shift
    local B="$1"; shift
    local csv="$1"; shift
    echo "=== scenario: $scenario (B=$B, K=64) ===" | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS \
        --paper-exact \
        --max-new 256 --repeat 1 --warmup \
        --scenarios "$scenario" --k-override 64 --b-override "$B" \
        --out "$csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (cell $scenario B=$B failed; continuing)" | tee -a "$LOG"
    echo | tee -a "$LOG"
}

run_cell multi_level_system    4 "$OUT_DIR/level_system.csv"
run_cell multi_chain_reasoning 4 "$OUT_DIR/chain_reasoning.csv"
run_cell multi_few_shot        4 "$OUT_DIR/few_shot.csv"

MERGED="$OUT_DIR/merged.csv"
echo "=== merging CSVs to $MERGED ===" | tee -a "$LOG"
first=1
for f in "$OUT_DIR/level_system.csv" "$OUT_DIR/chain_reasoning.csv" "$OUT_DIR/few_shot.csv"; do
    [[ -f "$f" ]] || { echo "WARN: $f missing" | tee -a "$LOG"; continue; }
    if [[ $first -eq 1 ]]; then
        cat "$f" > "$MERGED"; first=0
    else
        tail -n +2 "$f" >> "$MERGED"
    fi
done

{
    echo
    echo "=== merged results ==="
    [[ -f "$MERGED" ]] && cat "$MERGED" || echo "(no CSVs produced)"
    echo
    echo "Done. Results: $OUT_DIR"
} | tee -a "$LOG"

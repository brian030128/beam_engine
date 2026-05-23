#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp1c_70b_fp8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/paper_exp1c_70b_fp8_%j.out
#SBATCH --error=slurm/logs/paper_exp1c_70b_fp8_%j.err

# Picker-validity check (exp1c) on Llama-3-70B-FP8 TP=2.
#
# Mirrors scripts/paper-exp/exp1c_forced_2l.sbatch (1B/8B) but runs on
# RedHatAI/Meta-Llama-3-70B-Instruct-FP8 with the same cell shape as
# exp1a_70b_fp8_tp2.sbatch (K=64, B=4 across the 3 scenarios).
#
# At K=32/L_p=8192/B=8 the dispatch ablation (exp2_70b_fp8_tp2) found
# the free picker is 3.3% slower than forced 2L_1POOL after fp8
# recalibration. This script extends the check to the K=64 end-to-end
# cells used in the paper's headline 70B table:
#
#   bsk_2l_1p  → SHARED_2L_1POOL pinned (fused, prefill-only)
#   bsk_2l_dt  → SHARED_2L_DEC_TAIL pinned (root prefill + per-beam decode tail)
#
# Cells (all K=64, B=4 to match exp1a_70b_fp8_tp2):
#   multi_level_system
#   multi_chain_reasoning
#   multi_few_shot
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/forced_2l_70b_fp8/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# 70B FP8 weights are too large for the /work cache budget — use
# ephemeral SLURM_TMPDIR (see memory: feedback_large_model_ephemeral_cache).
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3   # fp8 KV cache (paired with L_kv-aware cost model)
export BS_KERNEL_TRACE_PLAN=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/forced_2l_70b_fp8"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp1c_70b_fp8.log"

{
    echo "=== exp1c TP=2 70B-FP8 forced-2L picker-validity started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=bf16)"
    echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
    echo
} | tee "$LOG"

# Picker + the two depth-2 forced variants. bench_sglang_e2e routes
# bsk_2l_1p / bsk_2l_dt through BsKernelBackend with
# available_strategies={SHARED_2L_1POOL} / {SHARED_2L_DEC_TAIL}.
METHODS="bs_kernel bsk_2l_1p bsk_2l_dt"

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
    echo "Compare decode_total_ms across method=bs_kernel / bsk_2l_1p / bsk_2l_dt per scenario."
} | tee -a "$LOG"

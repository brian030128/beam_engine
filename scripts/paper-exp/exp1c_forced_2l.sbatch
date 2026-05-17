#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp1c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/paper_exp1c_%j.out
#SBATCH --error=slurm/logs/paper_exp1c_%j.err

# Picker-vs-kernel sanity check for the End-to-end speedups table.
#
# In the K=32 sweep (see exp1b trace) the picker chose SHARED_2L_1POOL
# for 100% of steps, so the dec-tail kernel never contributed to the
# headline numbers. exp1a at K=64 should swing some cells toward
# SHARED_2L_DEC_TAIL; this script forces both depth-2 strategies on
# the same cells so we can verify the picker is making the right call:
#
#   bsk_2l_1p  → SHARED_2L_1POOL pinned (fused, prefill-only)
#   bsk_2l_dt  → SHARED_2L_DEC_TAIL pinned (root prefill + per-beam decode tail)
#
# If the picker's pick beats both forced variants where they overlap
# (and matches whichever it pinned), the picker is doing the right
# thing. If a forced variant beats the picker on a cell, the
# coefficients need re-autotuning.
#
# Cells mirror exp1a (K=64 override across all; 8B B halved to fit 80GB):
#   scenario              | model      | (B, K)
#   ----------------------|------------|--------
#   multi_level_system    | 1B / 8B    | (4, 64)
#   multi_chain_reasoning | 1B        | (32, 64)
#   multi_chain_reasoning | 8B        | (8, 64)
#   multi_few_shot        | 1B        | (32, 64)
#   multi_few_shot        | 8B        | (8, 64)
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/forced_2l/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/forced_2l"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp1c.log"

{
    echo "=== exp1c forced-2L comparison started $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

# Two forced variants + the picker for direct delta. bench_sglang_e2e
# routes bsk_2l_1p / bsk_2l_dt through BsKernelBackend with
# available_strategies={SHARED_2L_1POOL} / {SHARED_2L_DEC_TAIL} (see
# benchmarks/bs_kernel/bench_sglang_e2e.py:51-77).
METHODS="bs_kernel bsk_2l_1p bsk_2l_dt"

run_cell () {
    # run_cell <model_tag> <out_csv> <extra-args...>
    local model_tag="$1"; shift
    local csv="$1"; shift
    export BE_MODEL="meta-llama/${model_tag}"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS \
        --paper-exact \
        --max-new 256 --repeat 1 --warmup \
        --out "$csv" \
        "$@" \
        2>&1 | tee -a "$LOG"
}

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"
    csv_a="$OUT_DIR/${model_tag}_level_system.csv"
    echo "  multi_level_system  (B=4, K=64)" | tee -a "$LOG"
    run_cell "$model_tag" "$csv_a" \
        --scenarios multi_level_system --k-override 64

    if [[ "$model_tag" == "Llama-3.2-1B" ]]; then
        chain_B=32; few_shot_B=32
    else
        chain_B=8;  few_shot_B=8
    fi

    csv_b="$OUT_DIR/${model_tag}_chain_reasoning.csv"
    echo "  multi_chain_reasoning  (B=${chain_B}, K=64)" | tee -a "$LOG"
    run_cell "$model_tag" "$csv_b" \
        --scenarios multi_chain_reasoning --k-override 64 --b-override "$chain_B"

    csv_c="$OUT_DIR/${model_tag}_few_shot.csv"
    echo "  multi_few_shot  (B=${few_shot_B}, K=64)" | tee -a "$LOG"
    run_cell "$model_tag" "$csv_c" \
        --scenarios multi_few_shot --k-override 64 --b-override "$few_shot_B"

    merged="$OUT_DIR/${model_tag}.csv"
    cp "$csv_a" "$merged"
    [[ -f "$csv_b" ]] && tail -n +2 "$csv_b" >> "$merged"
    [[ -f "$csv_c" ]] && tail -n +2 "$csv_c" >> "$merged"
    echo "  merged → $merged" | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "Done. Results: $OUT_DIR" | tee -a "$LOG"
echo "Compare decode_total_ms across method=bs_kernel / bsk_2l_1p / bsk_2l_dt per (scenario, model)." | tee -a "$LOG"

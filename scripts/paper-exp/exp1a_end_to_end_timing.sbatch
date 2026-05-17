#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp1a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/paper_exp1a_%j.out
#SBATCH --error=slurm/logs/paper_exp1a_%j.err

# Reproduces docs/paper_story.md → "End-to-end speedups" timing table.
#
# Six cells (three scenarios × two Llama sizes), four methods
# (bs_kernel / fasttree / paged / mlca), warmup + one measured run.
# DeFT is in exp1a_deft.sbatch — separated because the vendored DeFT
# kernel is ~20-30x slower per step than the others and would push
# the job over the 60-min wallclock cap on the dev partition. Both
# sbatches write into the same OUT_DIR; the deft variant appends
# its rows to the existing per-model merged CSVs.
# One run is enough because each cell already averages over hundreds
# to thousands of beams × 256 decode steps — timer noise is dominated
# by the per-run integration. No trace flags — these would inflate
# plan_total_ms via the extra torch.cuda.synchronize() calls.
# Strategy-mix data comes from exp1b_end_to_end_strategy.sbatch.
#
# K=64 across all cells: at K=32 the picker chose SHARED_2L_1POOL for
# 100% of steps (see exp1b trace), which made the dec-tail kernel's
# contribution invisible in the paper table. K=64 doubles the per-beam
# tail workload (decode kernel rows) relative to the root tile, which
# is where DEC_TAIL is supposed to win. The forced-2L comparison is in
# exp1c_forced_2l.sbatch.
#
# Cells (all overridden to K=64):
#   scenario              | model      | (B, K)
#   ----------------------|------------|--------
#   multi_level_system    | 1B / 8B    | (4, 64)
#   multi_chain_reasoning | 1B        | (32, 64)
#   multi_chain_reasoning | 8B        | (8, 64)   [B halved; B=32 needs 90 GiB]
#   multi_few_shot        | 1B        | (32, 64)
#   multi_few_shot        | 8B        | (8, 64)   [B halved; B=16 needs 65 GiB]
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp1a.log"

{
    echo "=== exp1a end-to-end timing started $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

METHODS_ALL="fasttree bs_kernel paged mlca"
METHODS_NON_MLCA="fasttree bs_kernel paged"

run_cell () {
    # run_cell <model_tag> <methods> <out_csv> <extra-args...>
    local model_tag="$1"; shift
    local methods="$1"; shift
    local csv="$1"; shift
    export BE_MODEL="meta-llama/${model_tag}"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $methods \
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
    run_cell "$model_tag" "$METHODS_ALL" "$csv_a" \
        --scenarios multi_level_system --k-override 64

    if [[ "$model_tag" == "Llama-3.2-1B" ]]; then
        chain_B=32; few_shot_B=32
    else
        chain_B=8;  few_shot_B=8
    fi

    csv_b="$OUT_DIR/${model_tag}_chain_reasoning.csv"
    echo "  multi_chain_reasoning  (B=${chain_B}, K=64)  [no mlca; runs separately]" | tee -a "$LOG"
    run_cell "$model_tag" "$METHODS_NON_MLCA" "$csv_b" \
        --scenarios multi_chain_reasoning --k-override 64 --b-override "$chain_B"

    csv_b2="$OUT_DIR/${model_tag}_chain_reasoning_mlca.csv"
    echo "  multi_chain_reasoning  (mlca, may CUDA-fail)" | tee -a "$LOG"
    BE_MODEL="meta-llama/${model_tag}" uv run python \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods mlca --paper-exact \
        --max-new 256 --repeat 1 --warmup \
        --scenarios multi_chain_reasoning --k-override 64 --b-override "$chain_B" \
        --out "$csv_b2" \
        2>&1 | tee -a "$LOG" \
        || echo "  (mlca chain_reasoning failed; continuing)" | tee -a "$LOG"

    csv_c="$OUT_DIR/${model_tag}_few_shot.csv"
    echo "  multi_few_shot  (B=${few_shot_B}, K=64)" | tee -a "$LOG"
    run_cell "$model_tag" "$METHODS_ALL" "$csv_c" \
        --scenarios multi_few_shot --k-override 64 --b-override "$few_shot_B"

    merged="$OUT_DIR/${model_tag}.csv"
    cp "$csv_a" "$merged"
    [[ -f "$csv_b"  ]] && tail -n +2 "$csv_b"  >> "$merged"
    [[ -f "$csv_b2" ]] && tail -n +2 "$csv_b2" >> "$merged"
    [[ -f "$csv_c"  ]] && tail -n +2 "$csv_c"  >> "$merged"
    echo "  merged → $merged" | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "Done. Results: $OUT_DIR" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp1a_deft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/paper_exp1a_deft_%j.out
#SBATCH --error=slurm/logs/paper_exp1a_deft_%j.err

# Companion to exp1a_end_to_end_timing.sbatch — runs DeFT only on the
# same six K=64 cells and appends rows into the existing per-model
# merged CSVs (Llama-3.2-1B.csv / Llama-3.1-8B.csv).
#
# DeFT ships separately because its kernel is ~20-30x slower per step
# than the other four methods (smoke test at K=64/L_p=512 measured
# ~150 ms/step vs ~6 ms for FastTree); folding it into the main exp1a
# would push the job over the 60-min dev-partition wallclock cap.
#
# Cells mirror exp1a_end_to_end_timing.sbatch (K=64 override across
# all; 8B B halved to fit 80GB):
#   scenario              | model      | (B, K)
#   ----------------------|------------|--------
#   multi_level_system    | 1B / 8B    | (4, 64)
#   multi_chain_reasoning | 1B        | (32, 64)
#   multi_chain_reasoning | 8B        | (8, 64)
#   multi_few_shot        | 1B        | (32, 64)
#   multi_few_shot        | 8B        | (8, 64)
#
# Output:
#   benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing/
#     ${model_tag}_${scenario}_deft.csv  (raw per-cell)
#   appended into the same dir's existing ${model_tag}.csv per-model
#   merged file (header skipped).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp1a_deft.log"

{
    echo "=== exp1a DeFT-only run $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

run_cell () {
    # run_cell <model_tag> <out_csv> <extra-args...>
    local model_tag="$1"; shift
    local csv="$1"; shift
    export BE_MODEL="meta-llama/${model_tag}"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods deft \
        --paper-exact \
        --max-new 256 --repeat 1 --warmup \
        --out "$csv" \
        "$@" \
        2>&1 | tee -a "$LOG"
}

append_into_merged () {
    # append_into_merged <model_tag> <cell_csv>
    local model_tag="$1"
    local cell_csv="$2"
    local merged="$OUT_DIR/${model_tag}.csv"
    if [[ -f "$cell_csv" && -f "$merged" ]]; then
        tail -n +2 "$cell_csv" >> "$merged"
        echo "  appended → $merged" | tee -a "$LOG"
    elif [[ ! -f "$merged" ]]; then
        echo "  WARN: $merged missing (run exp1a_end_to_end_timing.sbatch first)" | tee -a "$LOG"
    else
        echo "  WARN: $cell_csv missing (deft run failed?)" | tee -a "$LOG"
    fi
}

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"

    csv_a="$OUT_DIR/${model_tag}_level_system_deft.csv"
    echo "  multi_level_system  (B=4, K=64)" | tee -a "$LOG"
    run_cell "$model_tag" "$csv_a" \
        --scenarios multi_level_system --k-override 64
    append_into_merged "$model_tag" "$csv_a"

    if [[ "$model_tag" == "Llama-3.2-1B" ]]; then
        chain_B=32; few_shot_B=32
    else
        chain_B=8;  few_shot_B=8
    fi

    csv_b="$OUT_DIR/${model_tag}_chain_reasoning_deft.csv"
    echo "  multi_chain_reasoning  (B=${chain_B}, K=64)" | tee -a "$LOG"
    run_cell "$model_tag" "$csv_b" \
        --scenarios multi_chain_reasoning --k-override 64 --b-override "$chain_B"
    append_into_merged "$model_tag" "$csv_b"

    csv_c="$OUT_DIR/${model_tag}_few_shot_deft.csv"
    echo "  multi_few_shot  (B=${few_shot_B}, K=64)" | tee -a "$LOG"
    run_cell "$model_tag" "$csv_c" \
        --scenarios multi_few_shot --k-override 64 --b-override "$few_shot_B"
    append_into_merged "$model_tag" "$csv_c"

    echo | tee -a "$LOG"
done

echo "Done. Results: $OUT_DIR" | tee -a "$LOG"

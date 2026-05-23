#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_exp1a_dft_q3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/paper_exp1a_deft_qwen3_%j.out
#SBATCH --error=slurm/logs/paper_exp1a_deft_qwen3_%j.err

# Qwen3-4B DeFT companion — mirrors exp1a_deft.sbatch on the same six
# K=64 cells but for the Qwen3-4B model only. Appends rows into the
# Qwen3-4B.csv merged file produced by exp1a_end_to_end_timing_qwen3_4b.sbatch
# (run that first).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL="Qwen/Qwen3-4B"

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp1a_deft_qwen3.log"

{
    echo "=== exp1a DeFT Qwen3-4B run $(date -Iseconds) ==="
    echo "model: $BE_MODEL"
    echo "HF_HOME: $HF_HOME"
    echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

model_tag="Qwen3-4B"
chain_B=8
few_shot_B=8

run_cell () {
    local csv="$1"; shift
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods deft \
        --paper-exact \
        --max-new 256 --repeat 1 --warmup \
        --out "$csv" \
        "$@" \
        2>&1 | tee -a "$LOG"
}

append_into_merged () {
    local cell_csv="$1"
    local merged="$OUT_DIR/${model_tag}.csv"
    if [[ -f "$cell_csv" && -f "$merged" ]]; then
        tail -n +2 "$cell_csv" >> "$merged"
        echo "  appended → $merged" | tee -a "$LOG"
    elif [[ ! -f "$merged" ]]; then
        echo "  WARN: $merged missing (run exp1a_end_to_end_timing_qwen3_4b.sbatch first)" | tee -a "$LOG"
    else
        echo "  WARN: $cell_csv missing (deft run failed?)" | tee -a "$LOG"
    fi
}

echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"

csv_a="$OUT_DIR/${model_tag}_level_system_deft.csv"
echo "  multi_level_system  (B=4, K=64)" | tee -a "$LOG"
run_cell "$csv_a" --scenarios multi_level_system --k-override 64
append_into_merged "$csv_a"

csv_b="$OUT_DIR/${model_tag}_chain_reasoning_deft.csv"
echo "  multi_chain_reasoning  (B=${chain_B}, K=64)" | tee -a "$LOG"
run_cell "$csv_b" --scenarios multi_chain_reasoning --k-override 64 --b-override "$chain_B"
append_into_merged "$csv_b"

csv_c="$OUT_DIR/${model_tag}_few_shot_deft.csv"
echo "  multi_few_shot  (B=${few_shot_B}, K=64)" | tee -a "$LOG"
run_cell "$csv_c" --scenarios multi_few_shot --k-override 64 --b-override "$few_shot_B"
append_into_merged "$csv_c"

echo "Done. Results: $OUT_DIR" | tee -a "$LOG"

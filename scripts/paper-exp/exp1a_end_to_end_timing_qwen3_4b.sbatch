#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_exp1a_q3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/paper_exp1a_qwen3_%j.out
#SBATCH --error=slurm/logs/paper_exp1a_qwen3_%j.err

# Qwen3-4B companion to exp1a_end_to_end_timing.sbatch — same six K=64
# cells but only this one model. Split out from the main exp1a because
# adding a third model to that loop overflows the 60-min dev cap.
#
# Cells (B sized for 4B between 1B and 8B):
#   multi_level_system    | (B=4,  K=64)
#   multi_chain_reasoning | (B=16, K=64)
#   multi_few_shot        | (B=16, K=64)
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing/

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
LOG="$OUT_DIR/exp1a_qwen3.log"

{
    echo "=== exp1a Qwen3-4B run $(date -Iseconds) ==="
    echo "model: $BE_MODEL"
    echo "HF_HOME: $HF_HOME"
    echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

METHODS_ALL="fasttree bs_kernel paged mlca"
METHODS_NON_MLCA="fasttree bs_kernel paged"
model_tag="Qwen3-4B"
chain_B=8
few_shot_B=8

run_cell () {
    local methods="$1"; shift
    local csv="$1"; shift
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $methods \
        --paper-exact \
        --max-new 256 --repeat 1 --warmup \
        --out "$csv" \
        "$@" \
        2>&1 | tee -a "$LOG"
}

echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"

csv_a="$OUT_DIR/${model_tag}_level_system.csv"
echo "  multi_level_system  (B=4, K=64)" | tee -a "$LOG"
run_cell "$METHODS_ALL" "$csv_a" \
    --scenarios multi_level_system --k-override 64

csv_b="$OUT_DIR/${model_tag}_chain_reasoning.csv"
echo "  multi_chain_reasoning  (B=${chain_B}, K=64)  [no mlca; runs separately]" | tee -a "$LOG"
run_cell "$METHODS_NON_MLCA" "$csv_b" \
    --scenarios multi_chain_reasoning --k-override 64 --b-override "$chain_B"

csv_b2="$OUT_DIR/${model_tag}_chain_reasoning_mlca.csv"
echo "  multi_chain_reasoning  (mlca, may CUDA-fail)" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods mlca --paper-exact \
    --max-new 256 --repeat 1 --warmup \
    --scenarios multi_chain_reasoning --k-override 64 --b-override "$chain_B" \
    --out "$csv_b2" \
    2>&1 | tee -a "$LOG" \
    || echo "  (mlca chain_reasoning failed; continuing)" | tee -a "$LOG"

csv_c="$OUT_DIR/${model_tag}_few_shot.csv"
echo "  multi_few_shot  (B=${few_shot_B}, K=64)" | tee -a "$LOG"
run_cell "$METHODS_ALL" "$csv_c" \
    --scenarios multi_few_shot --k-override 64 --b-override "$few_shot_B"

merged="$OUT_DIR/${model_tag}.csv"
cp "$csv_a" "$merged"
[[ -f "$csv_b"  ]] && tail -n +2 "$csv_b"  >> "$merged"
[[ -f "$csv_b2" ]] && tail -n +2 "$csv_b2" >> "$merged"
[[ -f "$csv_c"  ]] && tail -n +2 "$csv_c"  >> "$merged"
echo "  merged → $merged" | tee -a "$LOG"

echo "Done. Results: $OUT_DIR" | tee -a "$LOG"

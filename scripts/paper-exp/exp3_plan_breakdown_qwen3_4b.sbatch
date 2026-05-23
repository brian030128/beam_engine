#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_exp3_q3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/paper_exp3_qwen3_%j.out
#SBATCH --error=slurm/logs/paper_exp3_qwen3_%j.err

# Qwen3-4B planning-overhead breakdown — companion to exp3_plan_breakdown.sbatch.
#   A. cross-cell — paper grid, bs_kernel/fasttree with trace flags
#   B. substep   — K=16, L_p=8192, B=16 supplementary cell

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BS_KERNEL_TRACE_PLAN=1
export FT_TRACE_PLAN=1
export BE_MODEL="Qwen/Qwen3-4B"

OUT_BASE="benchmarks/bs_kernel/results/paper-exp/exp3_plan_breakdown"
CROSS_DIR="$OUT_BASE/cross"
SUBQ_DIR="$OUT_BASE/substep_qwen3_4b"
mkdir -p "$CROSS_DIR" "$SUBQ_DIR" slurm/logs
LOG="$OUT_BASE/exp3_qwen3.log"
METHODS="bs_kernel fasttree"
model_tag="Qwen3-4B"
chain_B=8
few_shot_B=8

{
    echo "=== exp3 Qwen3-4B plan-overhead breakdown started $(date -Iseconds) ==="
    echo "model: $BE_MODEL"
    echo "HF_HOME: $HF_HOME"
    echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

echo "==================== Pass A: paper-grid plan/forward split ====================" | tee -a "$LOG"

run_cross () {
    local csv="$1"; shift
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS --paper-exact \
        --max-new 256 --repeat 1 --warmup \
        --out "$csv" \
        "$@" \
        2>&1 | tee -a "$LOG"
}

echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"

echo "  multi_level_system (B=4, K=64)" | tee -a "$LOG"
run_cross "$CROSS_DIR/${model_tag}_level_system.csv" \
    --scenarios multi_level_system --k-override 64

echo "  multi_chain_reasoning (B=${chain_B}, K=64)" | tee -a "$LOG"
run_cross "$CROSS_DIR/${model_tag}_chain_reasoning.csv" \
    --scenarios multi_chain_reasoning --k-override 64 --b-override "$chain_B"

echo "  multi_few_shot (B=${few_shot_B}, K=64)" | tee -a "$LOG"
run_cross "$CROSS_DIR/${model_tag}_few_shot.csv" \
    --scenarios multi_few_shot --k-override 64 --b-override "$few_shot_B"

echo "==================== Pass B: per-step sub-breakdown ====================" | tee -a "$LOG"
echo "supplementary cell (Qwen3-4B): K=16, L_p=8192, B=16, max_new=256" | tee -a "$LOG"

echo "--- Qwen3-4B fasttree (FT_TRACE_PLAN=1) ---" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 16 --L_p 8192 --B 16 --max_new 256 \
    --max_pages 16500 --methods fasttree \
    --out "$SUBQ_DIR/ft.csv" \
    2>&1 | tee -a "$LOG"

echo "--- Qwen3-4B bs_kernel (BS_KERNEL_TRACE_PLAN=1) ---" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 16 --L_p 8192 --B 16 --max_new 256 \
    --max_pages 16500 --methods bs_kernel \
    --out "$SUBQ_DIR/bsk.csv" \
    2>&1 | tee -a "$LOG"

echo "Done. Results: $OUT_BASE" | tee -a "$LOG"
echo "  - cross-cell CSVs:    $CROSS_DIR" | tee -a "$LOG"
echo "  - Qwen3 sub-breakdown: $SUBQ_DIR (phase tables inline in $LOG)" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/paper_exp3_%j.out
#SBATCH --error=slurm/logs/paper_exp3_%j.err

# Reproduces docs/paper_story.md → "Planning-overhead breakdown" subsection.
#
# Two passes:
#   A. cross-cell plan/forward split  — same 6 cells as exp1 but
#      restricted to {bs_kernel, fasttree} with both backends' trace
#      flags enabled (BS_KERNEL_TRACE_PLAN=1 FT_TRACE_PLAN=1).
#      --repeat 1 (the trace path adds torch.cuda.synchronize() calls
#      that inflate plan_total relative to the no-trace timing run).
#   B. per-step sub-breakdown  — the K=16/L_p=4096/B=32 supplementary
#      cell on both models. Trace prints mean/median/p90/p99/total per
#      plan sub-phase for both backends.
#
# Output:
#   benchmarks/bs_kernel/results/paper-exp/exp3_plan_breakdown/cross/
#   benchmarks/bs_kernel/results/paper-exp/exp3_plan_breakdown/substep_8b/
#   benchmarks/bs_kernel/results/paper-exp/exp3_plan_breakdown/substep_1b/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export FT_TRACE_PLAN=1

OUT_BASE="benchmarks/bs_kernel/results/paper-exp/exp3_plan_breakdown"
CROSS_DIR="$OUT_BASE/cross"
SUB8B_DIR="$OUT_BASE/substep_8b"
SUB1B_DIR="$OUT_BASE/substep_1b"
mkdir -p "$CROSS_DIR" "$SUB8B_DIR" "$SUB1B_DIR" slurm/logs
LOG="$OUT_BASE/exp3.log"
METHODS="bs_kernel fasttree"

{
    echo "=== exp3 plan-overhead breakdown started $(date -Iseconds) ==="
    echo "trace flags: BS_KERNEL_TRACE_PLAN=$BS_KERNEL_TRACE_PLAN FT_TRACE_PLAN=$FT_TRACE_PLAN"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

############################################
# Pass A — cross-cell paper grid
############################################
echo "==================== Pass A: paper-grid plan/forward split ====================" | tee -a "$LOG"

run_cross () {
    local model_tag="$1"; shift
    local csv="$1"; shift
    export BE_MODEL="meta-llama/${model_tag}"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS --paper-exact \
        --max-new 256 --repeat 1 --warmup \
        --out "$csv" \
        "$@" \
        2>&1 | tee -a "$LOG"
}

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"

    if [[ "$model_tag" == "Llama-3.2-1B" ]]; then
        chain_B=32; few_shot_B=32
    else
        chain_B=8;  few_shot_B=8
    fi

    echo "  multi_level_system (B=4, K=64)" | tee -a "$LOG"
    run_cross "$model_tag" "$CROSS_DIR/${model_tag}_level_system.csv" \
        --scenarios multi_level_system --k-override 64

    echo "  multi_chain_reasoning (B=${chain_B}, K=64)" | tee -a "$LOG"
    run_cross "$model_tag" "$CROSS_DIR/${model_tag}_chain_reasoning.csv" \
        --scenarios multi_chain_reasoning --k-override 64 --b-override "$chain_B"

    echo "  multi_few_shot  (B=${few_shot_B}, K=64)" | tee -a "$LOG"
    run_cross "$model_tag" "$CROSS_DIR/${model_tag}_few_shot.csv" \
        --scenarios multi_few_shot --k-override 64 --b-override "$few_shot_B"
done

############################################
# Pass B — per-step sub-breakdown at supplementary cell
############################################
echo "==================== Pass B: per-step sub-breakdown ====================" | tee -a "$LOG"
echo "supplementary cell: K=16, L_p=4096, B=32, max_new=256 (8B variant uses K=16, L_p=4096, B=32; 1B variant uses K=32, L_p=8192, B=32)" | tee -a "$LOG"

# 8B sub-breakdown cell — matches slurm/trace_beam_search_plan_8b.sbatch
export BE_MODEL=meta-llama/Llama-3.1-8B
echo "--- 8B fasttree (FT_TRACE_PLAN=1) ---" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 16 --L_p 4096 --B 32 --max_new 256 \
    --max_pages 16500 --methods fasttree \
    --out "$SUB8B_DIR/ft.csv" \
    2>&1 | tee -a "$LOG"
echo "--- 8B bs_kernel (BS_KERNEL_TRACE_PLAN=1) ---" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 16 --L_p 4096 --B 32 --max_new 256 \
    --max_pages 16500 --methods bs_kernel \
    --out "$SUB8B_DIR/bsk.csv" \
    2>&1 | tee -a "$LOG"

# 1B sub-breakdown cell — matches slurm/trace_beam_search_plan_1b.sbatch
export BE_MODEL=meta-llama/Llama-3.2-1B
echo "--- 1B bs_kernel (BS_KERNEL_TRACE_PLAN=1) ---" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 32 --L_p 8192 --B 32 --max_new 256 \
    --max_pages 36500 --methods bs_kernel \
    --out "$SUB1B_DIR/bsk.csv" \
    2>&1 | tee -a "$LOG"
echo "--- 1B fasttree (FT_TRACE_PLAN=1) ---" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 32 --L_p 8192 --B 32 --max_new 256 \
    --max_pages 36500 --methods fasttree \
    --out "$SUB1B_DIR/ft.csv" \
    2>&1 | tee -a "$LOG"

echo "Done. Results: $OUT_BASE" | tee -a "$LOG"
echo "  - cross-cell CSVs:    $CROSS_DIR" | tee -a "$LOG"
echo "  - 8B sub-breakdown:   $SUB8B_DIR (phase tables inline in $LOG)" | tee -a "$LOG"
echo "  - 1B sub-breakdown:   $SUB1B_DIR (phase tables inline in $LOG)" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp3_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/paper_exp3_all_%j.out
#SBATCH --error=slurm/logs/paper_exp3_all_%j.err

# Cross-method plan-time breakdown (build_indices vs dispatch_decision).
#
# Five methods × six paper cells. dispatch_total_ms in the output CSV
# is populated when the relevant trace flag is on:
#   fasttree   → FT_TRACE_PLAN=1 → sum of per-step 'dispatch_ms'
#                (heuristic + parallelism convergence loop, mostly
#                amortised by the heuristic cache)
#   bs_kernel  → BS_KERNEL_TRACE_PLAN=1 → sum of per-step 'pick_ms'
#                (cost_model.pick_strategy_batch evaluations)
#   paged / deft / mlca → 0 (no dispatch decision in the plan path);
#                their plan time is entirely build_indices.
#
# Caveat: the trace flags add torch.cuda.synchronize() calls, which
# inflate the absolute plan totals here vs the no-trace timing path
# (exp1a). The meaningful number is the build_indices / dispatch
# *ratio*, not the absolute totals. We run all methods with the same
# trace setup so the comparison is fair across methods.
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp3_plan_breakdown/all/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export FT_TRACE_PLAN=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp3_plan_breakdown/all"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp3_all.log"

{
    echo "=== exp3 cross-method plan-time breakdown $(date -Iseconds) ==="
    echo "trace flags: BS_KERNEL_TRACE_PLAN=$BS_KERNEL_TRACE_PLAN FT_TRACE_PLAN=$FT_TRACE_PLAN"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

# Methods grouped by speed: paged/fasttree/bs_kernel are fast; mlca is
# slower; deft is slowest. We use the same methods list for all cells
# but tolerate per-method failures via --repeat 1 + the bench's existing
# per-(method, cell) try/except.
METHODS="paged fasttree mlca bs_kernel deft"

run_cell () {
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
    run_cell "$model_tag" "$OUT_DIR/${model_tag}_level_system.csv" \
        --scenarios multi_level_system --k-override 64

    echo "  multi_chain_reasoning (B=${chain_B}, K=64)" | tee -a "$LOG"
    run_cell "$model_tag" "$OUT_DIR/${model_tag}_chain_reasoning.csv" \
        --scenarios multi_chain_reasoning --k-override 64 --b-override "$chain_B"

    echo "  multi_few_shot  (B=${few_shot_B}, K=64)" | tee -a "$LOG"
    run_cell "$model_tag" "$OUT_DIR/${model_tag}_few_shot.csv" \
        --scenarios multi_few_shot --k-override 64 --b-override "$few_shot_B"
done

# Build a flat comparison table across all (model, scenario, method).
echo | tee -a "$LOG"
echo "=== cross-method comparison ===" | tee -a "$LOG"
{
    printf "%-12s %-22s %-10s %12s %12s %12s\n" \
        "model" "scenario" "method" "plan_total" "dispatch" "build_indices"
    for csv in "$OUT_DIR"/*.csv; do
        [[ -f "$csv" ]] || continue
        tag=$(basename "$csv" .csv)
        awk -F, -v tag="$tag" '
            NR==1 {
                for (i=1; i<=NF; i++) col[$i]=i
                next
            }
            {
                printf "%-12s %-22s %-10s %12.1f %12.1f %12.1f\n",
                    tag, $col["scenario"], $col["method"],
                    $col["plan_total_ms"], $col["dispatch_total_ms"],
                    $col["plan_total_ms"] - $col["dispatch_total_ms"]
            }' "$csv"
    done
} | tee -a "$LOG"

echo "Done. Results: $OUT_DIR" | tee -a "$LOG"

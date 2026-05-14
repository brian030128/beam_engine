#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=plan_trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_paper_table_plan_trace_%j.out
#SBATCH --error=slurm/logs/bench_paper_table_plan_trace_%j.err

# Plan-substep trace variant of bench_paper_table_h100.sbatch.
#
# Differences vs the paper-table run:
#   - methods restricted to bs_kernel + fasttree
#   - BS_KERNEL_TRACE_PLAN=1 / FT_TRACE_PLAN=1 enabled → per-substep
#     plan time summary printed to the log at the end of each
#     beam_search call (mean / median / p90 / p99 / total per phase)
#   - --repeat 1 (we want the structural breakdown, not 3-run timing
#     stability — the trace dumps adds torch.cuda.synchronize() calls
#     anyway so plan_total_ms in the CSV is inflated relative to the
#     no-trace run)
#
# Output: substep tables in slurm/logs/...out + the bench.log under the
# per-run results dir.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Trace env vars — both backends print a substep summary at the end of
# each beam_search call when these are set.
export BS_KERNEL_TRACE_PLAN=1
export FT_TRACE_PLAN=1

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/paper_table_plan_trace-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

METHODS="bs_kernel fasttree"

{
    echo "=== paper_table_plan_trace started $(date -Iseconds) ==="
    echo "methods:  ${METHODS}"
    echo "models:   Llama-3.2-1B, Llama-3.1-8B"
    echo "max_new=256, --warmup --repeat 1"
    echo "trace:    BS_KERNEL_TRACE_PLAN=${BS_KERNEL_TRACE_PLAN} FT_TRACE_PLAN=${FT_TRACE_PLAN}"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

run_cell () {
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

    # Pass 1: multi_level_system (natural B=4, K=32)
    csv_a="$RESULTS_DIR/${model_tag}_level_system.csv"
    echo "  pass 1: multi_level_system  (B=4, K=32 natural)" | tee -a "$LOG"
    run_cell "$model_tag" "$csv_a" \
        --scenarios multi_level_system

    # Pass 2: multi_chain_reasoning at K=16 (natural B=32)
    csv_b="$RESULTS_DIR/${model_tag}_chain_reasoning.csv"
    echo "  pass 2: multi_chain_reasoning  (B=32, K=16)" | tee -a "$LOG"
    run_cell "$model_tag" "$csv_b" \
        --scenarios multi_chain_reasoning \
        --k-override 16

    # Pass 3: multi_few_shot at (B=32, K=16) for 1B, (B=16, K=16) for 8B
    if [[ "$model_tag" == "Llama-3.2-1B" ]]; then
        few_shot_B=32
    else
        few_shot_B=16
    fi
    csv_c="$RESULTS_DIR/${model_tag}_few_shot.csv"
    echo "  pass 3: multi_few_shot  (B=${few_shot_B}, K=16)" | tee -a "$LOG"
    run_cell "$model_tag" "$csv_c" \
        --scenarios multi_few_shot \
        --b-override "$few_shot_B"

    echo | tee -a "$LOG"
done

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"
echo "Plan-trace substep tables are interleaved in: $LOG" | tee -a "$LOG"

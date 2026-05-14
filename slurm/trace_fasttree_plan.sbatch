#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=ft_trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/trace_fasttree_plan_%j.out
#SBATCH --error=slurm/logs/trace_fasttree_plan_%j.err

# Per-sub-phase trace of fasttree's plan_decode_step on the 3
# paper-exact scenarios. Confirms whether _build_metadata (meta_ms),
# which mirrors FastTree's fasttree_preparation() in the submodule, is
# the dominant CPU plan-time cost.
#
# FT_TRACE_PLAN=1 prints (gather, radix, partial, slots, meta, write)
# per-step phase timings at the end of each scenario.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/ft_plan_trace-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/trace.log"

{
    echo "=== fasttree plan-trace started $(date -Iseconds) ==="
    echo "scenarios: multi_level_system + multi_few_shot + multi_chain_reasoning (paper-exact)"
    echo "methods:   fasttree only"
    echo "models:    Llama-3.2-1B, Llama-3.1-8B"
    echo "FT_TRACE_PLAN=1  --warmup --repeat 1 (1 warm iter only, trace adds sync overhead)"
    echo
} | tee "$LOG"

# multi_few_shot uses B=32 on 1B / B=16 on 8B (matches the paper-ready
# report). multi_chain_reasoning uses K=16.
for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    export BE_MODEL="meta-llama/${model_tag}"
    export FT_TRACE_PLAN=1
    if [[ "$model_tag" == "Llama-3.2-1B" ]]; then
        FS_B=32
    else
        FS_B=16
    fi
    echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"

    echo "  multi_level_system (B=4, K=32)" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_level_system --methods fasttree \
        --paper-exact --max-new 256 --repeat 1 --warmup \
        --out "$RESULTS_DIR/${model_tag}_mls.csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"

    echo "  multi_few_shot (B=${FS_B}, K=16)" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_few_shot --methods fasttree \
        --paper-exact --b-override $FS_B \
        --max-new 256 --repeat 1 --warmup \
        --out "$RESULTS_DIR/${model_tag}_mfs.csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"

    echo "  multi_chain_reasoning (B=32, K=16)" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_chain_reasoning --methods fasttree \
        --paper-exact --k-override 16 \
        --max-new 256 --repeat 1 --warmup \
        --out "$RESULTS_DIR/${model_tag}_mcr.csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

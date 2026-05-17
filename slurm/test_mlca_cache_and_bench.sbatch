#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=mlca_cache_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/mlca_cache_test_%j.out
#SBATCH --error=slurm/logs/mlca_cache_test_%j.err

# 1) Correctness test (cached vs uncached vs paged ref).
# 2) Plan-phase microbench breakdown (with cache).
# 3) End-to-end re-bench on the four MLCA-supporting paper cells +
#    the two formerly-CUDA-faulting chain_reasoning cells.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGDIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing"
mkdir -p "$LOGDIR" slurm/logs
LOG="$LOGDIR/mlca_with_cache.log"

{
    echo "=== mlca + plan-cache run $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

echo "=== 1) Correctness test ===" | tee -a "$LOG"
uv run python tests/test_mlca_plan_cache.py 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== 2) Plan-phase microbench (after cache) ===" | tee -a "$LOG"
for cell in 1B-fewshot 1B-chain 1B-levelsys 8B-fewshot 8B-chain 8B-levelsys; do
    echo "--- $cell ---" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_mlca_plan_breakdown.py \
        --cell "$cell" --iters 50 2>&1 | tee -a "$LOG"
done

echo | tee -a "$LOG"
echo "=== 3) End-to-end MLCA re-bench (all 6 cells) ===" | tee -a "$LOG"

run_cell () {
    local model_tag="$1"; shift
    local scen="$1"; shift
    local bval="$1"; shift
    local csv="$1"; shift
    export BE_MODEL="meta-llama/${model_tag}"
    echo "    [${model_tag} / ${scen} B=${bval}]" | tee -a "$LOG"
    BE_MODEL="meta-llama/${model_tag}" uv run python \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods mlca --paper-exact \
        --max-new 256 --repeat 1 --warmup \
        --scenarios "$scen" --k-override 64 --b-override "$bval" \
        --out "$csv" \
        2>&1 | tee -a "$LOG" \
        || echo "    (failed; continuing)" | tee -a "$LOG"
}

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    if [[ "$model_tag" == "Llama-3.2-1B" ]]; then
        chain_B=32; few_shot_B=32
    else
        chain_B=8;  few_shot_B=8
    fi
    run_cell "$model_tag" multi_level_system    4              "$LOGDIR/${model_tag}_level_system_mlca_cached.csv"
    run_cell "$model_tag" multi_chain_reasoning "$chain_B"     "$LOGDIR/${model_tag}_chain_reasoning_mlca_cached.csv"
    run_cell "$model_tag" multi_few_shot        "$few_shot_B"  "$LOGDIR/${model_tag}_few_shot_mlca_cached.csv"
done

echo "Done. Logs: $LOG" | tee -a "$LOG"

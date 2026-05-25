#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=mlca_smcache_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/mlca_smcache_grid_%j.out
#SBATCH --error=slurm/logs/mlca_smcache_grid_%j.err

# Re-run MLCA on all 6 paper-exp cells with the SM90 plan-state cache.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MLCA_PLAN_CACHE=1

LOGDIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing"
LOG="$LOGDIR/mlca_smcache_grid.log"

{
    echo "=== mlca SM90 cache grid $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

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
    run_cell "$model_tag" multi_level_system    4              "$LOGDIR/${model_tag}_level_system_mlca_smcache.csv"
    run_cell "$model_tag" multi_chain_reasoning "$chain_B"     "$LOGDIR/${model_tag}_chain_reasoning_mlca_smcache.csv"
    run_cell "$model_tag" multi_few_shot        "$few_shot_B"  "$LOGDIR/${model_tag}_few_shot_mlca_smcache.csv"
done

echo "Done." | tee -a "$LOG"

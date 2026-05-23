#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp3_ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_exp3_ft_%j.out
#SBATCH --error=slurm/logs/paper_exp3_ft_%j.err

# Targeted re-run of fasttree alone, with FT_TRACE_PLAN=1 and the new
# dispatch_ms separation in _build_metadata. exp1a / exp1a_deft already
# have plan_total_ms for all 5 methods, prior exp3.log already has
# bs_kernel's pick_ms, and paged/deft/mlca have dispatch_ms = 0 by
# construction — fasttree's dispatch split is the only missing piece.
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp3_plan_breakdown/ft_dispatch/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FT_TRACE_PLAN=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp3_plan_breakdown/ft_dispatch"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/ft_dispatch.log"

{
    echo "=== exp3 fasttree-only dispatch_ms re-run $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

run_cell () {
    local model_tag="$1"; shift
    local csv="$1"; shift
    export BE_MODEL="meta-llama/${model_tag}"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods fasttree --paper-exact \
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

    echo "  multi_few_shot (B=${few_shot_B}, K=64)" | tee -a "$LOG"
    run_cell "$model_tag" "$OUT_DIR/${model_tag}_few_shot.csv" \
        --scenarios multi_few_shot --k-override 64 --b-override "$few_shot_B"
done

echo "Done. Results: $OUT_DIR" | tee -a "$LOG"

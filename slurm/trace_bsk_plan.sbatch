#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bsk_trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/trace_bsk_plan_%j.out
#SBATCH --error=slurm/logs/trace_bsk_plan_%j.err

# Per-sub-phase trace of bs_kernel's plan_decode_step on the 3
# paper-exact scenarios at the same configs as the fasttree trace
# (slurm/trace_fasttree_plan.sbatch). Used to build the side-by-side
# plan-time breakdown vs fasttree.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bsk_plan_trace-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/trace.log"

{
    echo "=== bs_kernel plan-trace started $(date -Iseconds) ==="
    echo "BS_KERNEL_TRACE_PLAN=1  --warmup --repeat 1"
    echo
} | tee "$LOG"

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    export BE_MODEL="meta-llama/${model_tag}"
    export BS_KERNEL_TRACE_PLAN=1
    if [[ "$model_tag" == "Llama-3.2-1B" ]]; then
        FS_B=32
    else
        FS_B=16
    fi
    echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"

    echo "  multi_level_system (B=4, K=32)" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_level_system --methods bs_kernel \
        --paper-exact --max-new 256 --repeat 1 --warmup \
        --out "$RESULTS_DIR/${model_tag}_mls.csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"

    echo "  multi_few_shot (B=${FS_B}, K=16)" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_few_shot --methods bs_kernel \
        --paper-exact --b-override $FS_B \
        --max-new 256 --repeat 1 --warmup \
        --out "$RESULTS_DIR/${model_tag}_mfs.csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"

    echo "  multi_chain_reasoning (B=32, K=16)" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_chain_reasoning --methods bs_kernel \
        --paper-exact --k-override 16 \
        --max-new 256 --repeat 1 --warmup \
        --out "$RESULTS_DIR/${model_tag}_mcr.csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

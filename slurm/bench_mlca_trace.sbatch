#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=mlca_trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/mlca_trace_%j.out
#SBATCH --error=slurm/logs/mlca_trace_%j.err

# Per-phase MLCA plan trace on three paper cells (different B,L_s,L_p).
# Trace adds torch.cuda.synchronize() between phases — inflates plan_total,
# so don't use these numbers as headline figures, only for sub-phase
# breakdown.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MLCA_TRACE_PLAN=1

LOGDIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing"
LOG="$LOGDIR/mlca_trace.log"

{
    echo "=== mlca per-phase trace $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
} | tee "$LOG"

run_cell () {
    local model_tag="$1"; shift
    local scen="$1"; shift
    local bval="$1"; shift
    echo "=== ${model_tag} / ${scen} B=${bval} K=64 ===" | tee -a "$LOG"
    BE_MODEL="meta-llama/${model_tag}" uv run python \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods mlca --paper-exact \
        --max-new 256 --repeat 1 --warmup \
        --scenarios "$scen" --k-override 64 --b-override "$bval" \
        --out /tmp/_mlca_trace_throwaway.csv \
        2>&1 | tee -a "$LOG"
}

# B=32 (the most plan-stressful cells)
run_cell "Llama-3.2-1B" multi_few_shot         32
run_cell "Llama-3.2-1B" multi_chain_reasoning  32
# B=8 (8B-sized cell)
run_cell "Llama-3.1-8B" multi_few_shot         8
# B=4 (smallest)
run_cell "Llama-3.2-1B" multi_level_system     4

echo "Done." | tee -a "$LOG"

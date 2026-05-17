#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=allplan_trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/allplan_trace_%j.out
#SBATCH --error=slurm/logs/allplan_trace_%j.err

# Per-phase plan trace for paged / fasttree / mlca / bs_kernel on the
# same paper cell — 1B/multi_few_shot (B=32, K=64), the most plan-stressful
# cell. Trace adds torch.cuda.synchronize() calls so plan totals are
# inflated; use only for sub-phase ratios.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PAGED_TRACE_PLAN=1
export FT_TRACE_PLAN=1
export MLCA_TRACE_PLAN=1
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL=meta-llama/Llama-3.2-1B

LOGDIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing"
LOG="$LOGDIR/allplan_trace.log"

{
    echo "=== all-method per-phase plan trace $(date -Iseconds) ==="
    echo "cell: Llama-3.2-1B / multi_few_shot / B=32 K=64"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

for method in paged fasttree mlca bs_kernel; do
    echo "############ method: $method ############" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods "$method" --paper-exact \
        --max-new 256 --repeat 1 --warmup \
        --scenarios multi_few_shot --k-override 64 --b-override 32 \
        --out "/tmp/_trace_${method}.csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "Done." | tee -a "$LOG"

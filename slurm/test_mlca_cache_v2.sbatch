#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=mlca_cache_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/mlca_cache_v2_%j.out
#SBATCH --error=slurm/logs/mlca_cache_v2_%j.err

# Re-run after fixing:
#   - bench_mlca_plan_breakdown.py: handle new 6-tuple pack return
#   - test_mlca_plan_cache.py: long prompts to skip pre-existing
#     short-prompt CoW edge case

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

LOGDIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing"
LOG="$LOGDIR/mlca_cache_v2.log"

{
    echo "=== mlca-cache v2 $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

echo "=== 1) Correctness test (cached vs uncached vs paged ref) ===" | tee -a "$LOG"
uv run python tests/test_mlca_plan_cache.py 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== 2) Plan-phase microbench (with cache) ===" | tee -a "$LOG"
for cell in 1B-fewshot 1B-chain 1B-levelsys 8B-fewshot 8B-chain 8B-levelsys; do
    echo "--- $cell ---" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_mlca_plan_breakdown.py \
        --cell "$cell" --iters 50 2>&1 | tee -a "$LOG"
done

echo "Done." | tee -a "$LOG"

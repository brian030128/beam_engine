#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=partial_page_fix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/partial_page_fix_%j.out
#SBATCH --error=slurm/logs/partial_page_fix_%j.err

# Verify the fix to page_driver._beam_set_page for partial-last-page
# prompts:
#   1) MLCA correctness test (cached vs uncached vs paged ref) on long
#      prompts that previously triggered IndexError.
#   2) Existing baseline correctness suite (test_baselines.py) — paged
#      vs tree, including beam_width=1 greedy reproducibility.
#   3) Smoke-bench on one paper cell to confirm tree_driver path
#      untouched.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

LOGDIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing"
LOG="$LOGDIR/partial_page_fix.log"

{
    echo "=== partial-page fix verification $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

echo "=== 1) MLCA correctness (cached vs uncached vs paged) ===" | tee -a "$LOG"
uv run python tests/test_mlca_plan_cache.py 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== 2) Cross-baseline correctness suite (paged vs tree) ===" | tee -a "$LOG"
uv run python tests/test_baselines.py 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== 3) Bench smoke (paper-exact 1B/level_system MLCA) ===" | tee -a "$LOG"
BE_MODEL="meta-llama/Llama-3.2-1B" uv run python \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods mlca paged --paper-exact \
    --max-new 64 --repeat 1 --warmup \
    --scenarios multi_level_system --k-override 64 \
    --out "$LOGDIR/partial_page_fix_smoke.csv" \
    2>&1 | tee -a "$LOG"

echo "Done." | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=mlca_sm90_cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:35:00
#SBATCH --output=slurm/logs/mlca_sm90_cache_%j.out
#SBATCH --error=slurm/logs/mlca_sm90_cache_%j.err

# Verify the SM90 MLCA plan-state cache (in-place kv_len patch in
# _int_workspace_buffer). Three checks:
#   1) MLCA-cached vs MLCA-uncached tokens on long prompts.
#   2) MLCA-cached vs paged-attention reference.
#   3) End-to-end plan/step delta on 1B/multi_few_shot B=32 K=64.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGDIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing"
LOG="$LOGDIR/mlca_sm90_cache.log"

{
    echo "=== mlca SM90 cache verify $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

echo "=== 1) Correctness: cached vs uncached MLCA, both vs paged ===" | tee -a "$LOG"
MLCA_PLAN_CACHE=1 uv run python tests/test_mlca_plan_cache.py 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== 2a) Bench MLCA (cache OFF) on 1B/multi_few_shot B=32 K=64 ===" | tee -a "$LOG"
BE_MODEL="meta-llama/Llama-3.2-1B" MLCA_PLAN_CACHE=0 uv run python \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods mlca --paper-exact \
    --max-new 256 --repeat 1 --warmup \
    --scenarios multi_few_shot --k-override 64 --b-override 32 \
    --out "$LOGDIR/mlca_sm90_cache_off.csv" 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== 2b) Bench MLCA (cache ON) on 1B/multi_few_shot B=32 K=64 ===" | tee -a "$LOG"
BE_MODEL="meta-llama/Llama-3.2-1B" MLCA_PLAN_CACHE=1 uv run python \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods mlca --paper-exact \
    --max-new 256 --repeat 1 --warmup \
    --scenarios multi_few_shot --k-override 64 --b-override 32 \
    --out "$LOGDIR/mlca_sm90_cache_on.csv" 2>&1 | tee -a "$LOG"

echo "Done." | tee -a "$LOG"

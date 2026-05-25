#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=mlca_ld_smcache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/mlca_ld_smcache_%j.out
#SBATCH --error=slurm/logs/mlca_ld_smcache_%j.err

# MLCA at the exp4 long-decode cell with SM90 plan-state cache.
# (1B, K=32, L_p=8192, B=16, max_new=2048)

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B
export MLCA_PLAN_CACHE=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp4_long_decode"
LOG="$OUT_DIR/mlca_smcache.log"

{ echo "=== mlca long-decode SM90 cache $(date -Iseconds) ==="; } | tee "$LOG"

uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 32 --L_p 8192 --B 16 --max_new 2048 \
    --prompts-file data/gov_report.jsonl \
    --max_pages 80000 \
    --warmup \
    --methods mlca \
    --out "$OUT_DIR/mlca_smcache.csv" \
    2>&1 | tee -a "$LOG"
echo "Done." | tee -a "$LOG"

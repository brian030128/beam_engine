#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=gov_3l1p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/bench_gov_3l1p_force_%j.out
#SBATCH --error=slurm/logs/bench_gov_3l1p_force_%j.err

# Force bs_kernel to SHARED_3L_1POOL only — compare against default
# picker + forced 3L_DEC_TAIL at the long-decode cell.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B
export BS_KERNEL_TRACE_PLAN=1

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/gov_3l1p_force-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== gov_3l1p_force started $(date -Iseconds) ==="
    echo "cell:  L_p=8192, K=32, B=16, max_new=2048"
    echo
} | tee "$LOG"

echo "=== bs_kernel_3l1p (forced SHARED_3L_1POOL) ===" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 32 --L_p 8192 --B 16 --max_new 2048 \
    --prompts-file data/gov_report.jsonl \
    --max_pages 80000 \
    --warmup \
    --methods bs_kernel_3l1p \
    --out "$RESULTS_DIR/bs_kernel_3l1p.csv" \
    2>&1 | tee -a "$LOG"

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

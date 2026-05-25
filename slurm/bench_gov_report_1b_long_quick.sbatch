#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=gov_1b_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/bench_gov_report_1b_long_quick_%j.out
#SBATCH --error=slurm/logs/bench_gov_report_1b_long_quick_%j.err

# Quick sanity pass: bs_kernel + dbs_bs_kernel only at long-decode
# config, to verify (1) the new SHARED_3L_DEC_TAIL strategy gets picked,
# (2) the vectorized DBS top-K is much faster than before.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B
export BS_KERNEL_TRACE_PLAN=1

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/gov_report_1b_long_quick-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== gov_report_1b_long_quick started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL"
    echo "cell:     L_p=8192, K=32, B=16, max_new=2048"
    echo "methods:  bs_kernel dbs_bs_kernel  (single warm pass each)"
    echo
} | tee "$LOG"

for m in bs_kernel dbs_bs_kernel; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 32 --L_p 8192 --B 16 --max_new 2048 \
        --prompts-file data/gov_report.jsonl \
        --max_pages 80000 \
        --warmup \
        --methods "$m" \
        --out "$RESULTS_DIR/${m}.csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bs_trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/trace_beam_search_plan_8b_%j.out
#SBATCH --error=slurm/logs/trace_beam_search_plan_8b_%j.err

# Per-sub-phase plan trace on the 8B beam-search cell where bs_kernel's
# plan_total exceeds fasttree's:
#   L_p=4096, K=16, B=32, max_new=256.
# Confirms whether bs_kernel's larger plan time comes from cache
# invalidation due to top-K fork (cf the SGLang non-fork bench where
# bs_kernel plan is consistently smaller).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.1-8B

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bs_trace_8b-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/trace.log"

{
    echo "=== beam-search plan trace 8B started $(date -Iseconds) ==="
    echo "config: L_p=4096 K=16 B=32 max_new=256"
    echo
} | tee "$LOG"

echo "--- fasttree (FT_TRACE_PLAN=1) ---" | tee -a "$LOG"
FT_TRACE_PLAN=1 uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 16 --L_p 4096 --B 32 --max_new 256 \
    --max_pages 10500 --methods fasttree \
    --out "$RESULTS_DIR/ft.csv" \
    2>&1 | tee -a "$LOG"

echo
echo "--- bs_kernel (BS_KERNEL_TRACE_PLAN=1) ---" | tee -a "$LOG"
BS_KERNEL_TRACE_PLAN=1 uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 16 --L_p 4096 --B 32 --max_new 256 \
    --max_pages 10500 --methods bs_kernel \
    --out "$RESULTS_DIR/bsk.csv" \
    2>&1 | tee -a "$LOG"

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

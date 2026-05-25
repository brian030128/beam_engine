#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bs_tr_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/trace_beam_search_plan_1b_%j.out
#SBATCH --error=slurm/logs/trace_beam_search_plan_1b_%j.err

# bs_kernel + fasttree plan-time sub-phase trace on the 1B beam-search
# cell (L_p=8192, K=32, B=32, max_new=256) where bs_kernel wins
# decode_total decisively but its plan path is slightly heavier than
# fasttree's. Report which strategy bs_kernel's picker chooses.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bs_trace_1b-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/trace.log"

{
    echo "=== beam-search plan trace 1B started $(date -Iseconds) ==="
    echo "config: L_p=8192 K=32 B=32 max_new=256"
    echo
} | tee "$LOG"

echo "--- bs_kernel (BS_KERNEL_TRACE_PLAN=1) ---" | tee -a "$LOG"
BS_KERNEL_TRACE_PLAN=1 uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 32 --L_p 8192 --B 32 --max_new 256 \
    --max_pages 36500 --methods bs_kernel \
    --out "$RESULTS_DIR/bsk.csv" \
    2>&1 | tee -a "$LOG"

echo
echo "--- fasttree (FT_TRACE_PLAN=1) ---" | tee -a "$LOG"
FT_TRACE_PLAN=1 uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 32 --L_p 8192 --B 32 --max_new 256 \
    --max_pages 36500 --methods fasttree \
    --out "$RESULTS_DIR/ft.csv" \
    2>&1 | tee -a "$LOG"

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

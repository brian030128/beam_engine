#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_picker_last
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/bench_modes_picker_last_%j.out
#SBATCH --error=slurm/logs/bench_modes_picker_last_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_modes_picker_last-${TS}"
mkdir -p "$RESULTS_DIR"

# Same K=64/L_p=8192/B=32/max_new=256, but with picker LAST so its
# decode timings aren't contaminated by first-run JIT compilation
# overhead. If picker now matches forced DEC_TAIL within noise, the
# previous gap was first-run overhead. If gap persists, it's cost-
# model evaluation cost in pick_strategy_batch.
uv run python benchmarks/bs_kernel/bench_modes.py \
    --K 64 \
    --L_p 8192 \
    --B 32 \
    --max_new 256 \
    --modes SHARED_2L_DEC_TAIL SHARED_2L_1POOL SHARED_3L_1POOL SHARED_3L_DEC_TAIL picker \
    --out "$RESULTS_DIR/bench_modes_picker_last.csv" \
    2>&1 | tee "$RESULTS_DIR/bench_modes_picker_last.log"

echo
echo "Done. Results: $RESULTS_DIR"

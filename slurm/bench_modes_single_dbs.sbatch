#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_modes_dbs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/bench_modes_single_dbs_%j.out
#SBATCH --error=slurm/logs/bench_modes_single_dbs_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_modes_single_dbs-${TS}"
mkdir -p "$RESULTS_DIR"

# Single grid point: K=64 × L_p=8192 × B=32 × max_new=256, with diverse
# beam search wrapping each of bs_kernel's modes (picker auto + forced
# strategies). Picker placed last so its decode timing isn't
# contaminated by first-run JIT compilation overhead.
uv run python benchmarks/bs_kernel/bench_modes.py \
    --K 64 \
    --L_p 8192 \
    --B 32 \
    --max_new 256 \
    --dbs \
    --modes SHARED_2L_DEC_TAIL SHARED_2L_1POOL SHARED_3L_1POOL SHARED_3L_DEC_TAIL SHARED_2L_2POOL SHARED_3L_2POOL PER_BEAM picker \
    --out "$RESULTS_DIR/bench_modes_single_dbs.csv" \
    2>&1 | tee "$RESULTS_DIR/bench_modes_single_dbs.log"

echo
echo "Done. Results: $RESULTS_DIR"

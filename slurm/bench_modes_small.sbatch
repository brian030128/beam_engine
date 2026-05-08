#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_modes_small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/bench_modes_small_%j.out
#SBATCH --error=slurm/logs/bench_modes_small_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_modes_small-${TS}"
mkdir -p "$RESULTS_DIR"

# Small grid: K=4 × B=4 × L_p=8192 × max_new=256.
# K*B=16 beams (vs 2048 in the canonical cell). The per-beam tile waves
# fit in 1 SM-wave so SHARED's tile-padding cost is essentially free
# here — DEC_TAIL's launch+merge overhead may dominate.
# Picker last for fair JIT; both vanilla and DBS variants.

MODES_ORDERED="SHARED_2L_DEC_TAIL SHARED_2L_1POOL SHARED_3L_1POOL SHARED_3L_DEC_TAIL SHARED_2L_2POOL SHARED_3L_2POOL PER_BEAM picker"

echo "=== Vanilla beam search ==="
uv run python benchmarks/bs_kernel/bench_modes.py \
    --K 4 \
    --L_p 8192 \
    --B 4 \
    --max_new 256 \
    --modes $MODES_ORDERED \
    --out "$RESULTS_DIR/vanilla.csv" \
    2>&1 | tee "$RESULTS_DIR/vanilla.log"

echo
echo "=== Diverse beam search (G=4, λ=0.5) ==="
uv run python benchmarks/bs_kernel/bench_modes.py \
    --K 4 \
    --L_p 8192 \
    --B 4 \
    --max_new 256 \
    --modes $MODES_ORDERED \
    --dbs \
    --out "$RESULTS_DIR/dbs.csv" \
    2>&1 | tee "$RESULTS_DIR/dbs.log"

echo
echo "Done. Results: $RESULTS_DIR"

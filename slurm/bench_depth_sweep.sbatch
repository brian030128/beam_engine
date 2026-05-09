#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_depth_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_depth_sweep_%j.out
#SBATCH --error=slurm/logs/bench_depth_sweep_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# FlashInfer JIT bakes $cuda_home into build.ninja; ensure the compute
# node uses the loaded cuda/12.4 path, not the broken /usr/local/cuda.
export CUDA_HOME=/work/HPC_software/LMOD/nvidia/packages/cuda-12.4
export FLASHINFER_NVCC=$CUDA_HOME/bin/nvcc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_depth_sweep-${TS}"
mkdir -p "$RESULTS_DIR"

# Smoke + validation sweep for the depth ∈ {2..6} extension (Phase 0.5b).
# Confirms:
#   1. depth=4..6 wrappers construct + execute without crashing
#   2. picker emits deeper picks when allowed (max_dispatch_depth=6)
#   3. forced-depth modes match the picker's argmin within the allowed set
#
# Cells: K=64 (severe bimodal where 2-pool / DEC_TAIL has historically won)
#        L_p=8192 (long enough to amortize cascade plan)
#        B=8, 32 (mid + large batch, where the bimodal regime is sharpest)
#        max_new=64 (short enough to keep job under 30 min)

uv run python benchmarks/bs_kernel/bench_modes.py \
    --K 64 \
    --L_p 8192 \
    --B 8 32 \
    --max_new 64 \
    --max_dispatch_depth 6 \
    --max_cascade_levels 6 \
    --modes \
        picker \
        SHARED_2L_1POOL SHARED_2L_2POOL SHARED_2L_DEC_TAIL \
        SHARED_3L_1POOL SHARED_3L_2POOL SHARED_3L_DEC_TAIL \
        SHARED_4L_1POOL SHARED_4L_2POOL SHARED_4L_DEC_TAIL \
        SHARED_5L_1POOL SHARED_5L_2POOL SHARED_5L_DEC_TAIL \
        SHARED_6L_1POOL SHARED_6L_2POOL SHARED_6L_DEC_TAIL \
    --out "$RESULTS_DIR/bench_modes.csv" \
    2>&1 | tee "$RESULTS_DIR/bench_modes.log"

echo
echo "Done. Results: $RESULTS_DIR"

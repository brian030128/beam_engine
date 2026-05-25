#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=smoke_deft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/smoke_deft_%j.out
#SBATCH --error=slurm/logs/smoke_deft_%j.err

# Tiny DeftBackend smoke test: does the vendored DeFT kernel + our
# metadata builder produce a non-NaN, non-crashing forward pass?
#
# Two cells:
#   small : K=4,  L_p=128, B=2,  max_new=8   — single radix subtree per prompt
#   midK  : K=64, L_p=512, B=2,  max_new=8   — exercises BLOCK_M=32 chunking
#
# Also runs fasttree on the same cells for a side-by-side decode_total_ms
# baseline so we can see whether DeFT is in a reasonable ballpark or
# wildly off (e.g. 100x slower would suggest a metadata bug).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B

mkdir -p slurm/logs benchmarks/bs_kernel/results/smoke
LOG=benchmarks/bs_kernel/results/smoke/deft_smoke.log

{
    echo "=== deft smoke test $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
    echo "--- small cell (K=4, L_p=128, B=2, max_new=8) ---"
} | tee "$LOG"

uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 4 --L_p 128 --B 2 --max_new 8 \
    --methods deft fasttree \
    --out benchmarks/bs_kernel/results/smoke/deft_small.csv \
    2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "--- midK cell (K=64, L_p=512, B=2, max_new=8) — tests BLOCK_M chunking ---" | tee -a "$LOG"

uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 64 --L_p 512 --B 2 --max_new 8 \
    --methods deft fasttree \
    --out benchmarks/bs_kernel/results/smoke/deft_midk.csv \
    2>&1 | tee -a "$LOG"

echo "Done." | tee -a "$LOG"

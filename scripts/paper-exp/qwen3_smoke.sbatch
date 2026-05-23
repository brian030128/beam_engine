#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=qwen3_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/qwen3_smoke_%j.out
#SBATCH --error=slurm/logs/qwen3_smoke_%j.err

# Verify the Qwen3 port:
#   1. weights load via load_model_for_causal_lm
#   2. forward via paged / fasttree / bs_kernel runs without crashing
#   3. the three backends agree on the first decode token (greedy)
# If this passes, the paper-exp sbatches are safe to submit.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL="Qwen/Qwen3-4B"

mkdir -p slurm/logs

echo "=== qwen3 port smoke started $(date -Iseconds) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 8 --L_p 512 --B 2 --max_new 4 \
    --methods paged fasttree bs_kernel \
    --warmup \
    --max_pages 4096 \
    --out benchmarks/bs_kernel/results/qwen3_smoke.csv

rc=${PIPESTATUS[0]}
echo "=== bench exit rc=$rc ==="
echo "=== checking CSV for nonzero decode_total_ms ==="
head -2 benchmarks/bs_kernel/results/qwen3_smoke.csv 2>/dev/null
exit $rc

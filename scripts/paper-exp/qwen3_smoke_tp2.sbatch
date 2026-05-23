#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=qwen3_smoke_tp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/qwen3_smoke_tp2_%j.out
#SBATCH --error=slurm/logs/qwen3_smoke_tp2_%j.err

# Quick TP=2 smoke for the Qwen3 port — same shape as the single-GPU
# smoke but launched via torchrun --nproc_per_node=2.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL="Qwen/Qwen3-4B"

echo "=== qwen3 TP=2 smoke started $(date -Iseconds) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"

torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/bench_batched.py \
    --K 8 --L_p 512 --B 2 --max_new 4 \
    --methods paged fasttree bs_kernel \
    --warmup \
    --max_pages 4096 \
    --out benchmarks/bs_kernel/results/qwen3_smoke_tp2.csv

rc=${PIPESTATUS[0]}
echo "=== exit rc=$rc ==="
echo "=== CSV head ==="
head -4 benchmarks/bs_kernel/results/qwen3_smoke_tp2.csv 2>/dev/null
exit $rc

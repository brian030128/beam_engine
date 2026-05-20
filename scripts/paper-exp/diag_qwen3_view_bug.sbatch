#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=diag_q3_view
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/diag_qwen3_view_%j.out
#SBATCH --error=slurm/logs/diag_qwen3_view_%j.err

# Quick diagnostic — Qwen3-4B TP=2 with full traceback dumped on failure.
# Just multi_level_system, B=2, K=8, max_new=8.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL="Qwen/Qwen3-4B"
export BE_TRACEBACK_ON_FAIL=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/diag_qwen3_view"
mkdir -p "$OUT_DIR" slurm/logs

torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods bs_kernel \
    --paper-exact --no-stage2 \
    --max-new 8 --repeat 1 \
    --scenarios multi_level_system --k-override 8 --b-override 2 \
    --out "$OUT_DIR/probe.csv" \
    2>&1 | tee "$OUT_DIR/probe.log"

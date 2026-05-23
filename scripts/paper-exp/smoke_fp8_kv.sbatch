#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=smoke_fp8_kv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/smoke_fp8_kv_%j.out
#SBATCH --error=slurm/logs/smoke_fp8_kv_%j.err

# Smoke test: does fp8 KV cache work end-to-end on 8B (bf16 weights,
# fp8 KV)? Memory note says append_paged_kv_cache dispatch fails on
# uint8-as-fp8 — confirm or refute.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
unset HF_HOME
export BE_MODEL="meta-llama/Llama-3.1-8B"
export BE_DTYPE=fp16
export BE_KV_DTYPE=fp8_e4m3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/fp8_kv_smoke"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/smoke.log"

echo "=== fp8 KV smoke on 8B started $(date -Iseconds) ===" | tee "$LOG"

uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods bs_kernel paged --paper-exact \
    --scenarios multi_level_system --k-override 16 --b-override 2 \
    --max-new 16 --repeat 1 \
    --out "$OUT_DIR/smoke.csv" \
    2>&1 | tee -a "$LOG"

echo "Done." | tee -a "$LOG"

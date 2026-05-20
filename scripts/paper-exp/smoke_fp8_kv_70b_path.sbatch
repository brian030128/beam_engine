#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=smk_fp8kv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/smk_fp8kv_%j.out
#SBATCH --error=slurm/logs/smk_fp8kv_%j.err

# Quick fp8-KV smoke test on Llama-3.2-1B (single GPU, tiny shape).
# Validates the BE_KV_DTYPE=fp8_e4m3 path before the 70B job actually
# loads its multi-GB checkpoint.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Stay on the $HOME cache (already auth'd) — /work has no token.
export BE_MODEL="meta-llama/Llama-3.1-8B"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3
export BE_TRACEBACK_ON_FAIL=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/smoke_fp8_kv_70b_path"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/probe.log"

{
    echo "=== fp8-KV smoke on 1B started $(date -Iseconds) ==="
    echo "BE_DTYPE=$BE_DTYPE BE_KV_DTYPE=$BE_KV_DTYPE"
} | tee "$LOG"

uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods bs_kernel bsk_2l_1p bsk_2l_dt \
    --paper-exact --no-stage2 \
    --max-new 16 --repeat 1 --warmup \
    --scenarios multi_level_system --k-override 8 --b-override 2 \
    --out "$OUT_DIR/sglang.csv" \
    2>&1 | tee -a "$LOG" \
    || echo "  (sglang fp8-KV failed; see traceback above)" | tee -a "$LOG"

echo | tee -a "$LOG"
echo "--- beam_search via bench_batched.py ---" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 8 --L_p 2048 --B 2 --max_new 16 \
    --warmup --methods bs_kernel bs_kernel_2l1p bs_kernel_2ldt \
    --out "$OUT_DIR/batched.csv" \
    2>&1 | tee -a "$LOG" \
    || echo "  (batched fp8-KV failed; see traceback above)" | tee -a "$LOG"

echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_div1b_ctl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/exp2_divlevels_1b_control_%j.out
#SBATCH --error=slurm/logs/exp2_divlevels_1b_control_%j.err

# CONTROL for the divergence-level diagnostic: identical 1B gov_report
# beam cell, but with NO level threshold (defaults => legacy split-on-any-
# fork). Confirms 3L/4L/5L genuinely form here (isolating "threshold too
# strict" from "my gate over-rejects / a bug"). Short max_new=512 for speed.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME=/work/u4320956/hf
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B-Instruct
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16
export BE_FORCE_FA2=1
export BS_KERNEL_TRACE_PLAN=1
# NOTE: intentionally NOT setting BS_KERNEL_MIN_LEVEL_TOKENS/_CHILDREN
# (defaults: tokens=0, children=1 => legacy behavior).

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp2_divlevels_1b_control"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/control.log"
echo "=== control (no threshold) $(date -Iseconds) — expect 3L+ to form ===" | tee "$LOG"

for m in bs_kernel_3l1p bs_kernel_5l1p bs_kernel_5ldt; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 32 --L_p 8192 --B 8 --max_new 512 \
        --prompts-file data/gov_report.jsonl \
        --max_pages 20000 --warmup \
        --measure_start 400 --measure_end 512 \
        --methods "$m" --out "$OUT_DIR/${m}.csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done
echo "=== control done $(date -Iseconds) ===" | tee -a "$LOG"

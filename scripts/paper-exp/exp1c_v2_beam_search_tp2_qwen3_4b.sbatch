#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_bs_q3_tp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/paper_exp1c_v2_bs_qwen3_tp2_%j.out
#SBATCH --error=slurm/logs/paper_exp1c_v2_bs_qwen3_tp2_%j.err

# Beam_search-only resubmit for Qwen3-4B exp1c_v2 cell that OOM'd on
# single-rank bench_batched.py. bench_batched now calls init_tp() so
# torchrun --nproc_per_node=2 actually pins each rank to its own GPU.
#
# Cell:
#   beam_search (gov_report): K=32, L_p=8192, B=8, max_new=2048, tp=2.
# Output appended to the existing exp1c_v2/Qwen3-4B directory so the
# already-good sglang cells are untouched.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export HF_HOME=/work/u4320956/hf
export BE_MODEL="Qwen/Qwen3-4B"

MODEL_TAG="Qwen3-4B"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1c_v2/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run_beam_search_tp2.log"

{
    echo "=== exp1c_v2 ${MODEL_TAG} beam_search-only TP=2 started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2)"
    echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
    echo
} | tee "$LOG"

BATCHED_METHODS="bs_kernel bs_kernel_2l1p bs_kernel_2ldt"
K=32
L_p=8192
B=8
MAX_NEW=2048
CSV="$OUT_DIR/beam_search.csv"

{
    echo
    echo "=========================================================="
    echo "=== cell: beam_search  (K=${K}, L_p=${L_p}, B=${B}, max_new=${MAX_NEW}, tp=2) ==="
    echo "=========================================================="
} | tee -a "$LOG"

first=1
for m in $BATCHED_METHODS; do
    echo "  --- method: $m ---" | tee -a "$LOG"
    tmp="$OUT_DIR/beam_search_${m}.csv"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_batched.py \
        --K "$K" --L_p "$L_p" --B "$B" --max_new "$MAX_NEW" \
        --prompts-file data/gov_report.jsonl \
        --warmup \
        --methods "$m" \
        --out "$tmp" \
        2>&1 | tee -a "$LOG" \
        || { echo "  (beam_search $m failed; continuing)" | tee -a "$LOG"; continue; }
    if [[ $first -eq 1 ]]; then
        [[ -f "$tmp" ]] && cp "$tmp" "$CSV" && first=0
    else
        [[ -f "$tmp" ]] && tail -n +2 "$tmp" >> "$CSV"
    fi
done

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

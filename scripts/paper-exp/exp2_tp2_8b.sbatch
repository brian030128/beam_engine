#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp2_tp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/paper_exp2_tp2_%j.out
#SBATCH --error=slurm/logs/paper_exp2_tp2_%j.err

# TP=2 Llama-3.1-8B dispatch-space ablation (companion to exp2).
# Long-decode beam-search cell at K=32 L_p=8192 B=8 max_new=2048, with
# each forced-strategy variant pinned. B was reduced from 16 → 8 so the
# page-table eager allocation fits in 80 GiB H100 (per-rank). Matches
# 70B/TP=2 cell shape so plan-time numbers compare directly.
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp2_dispatch_ablation_tp2_8b/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.1-8B
export BS_KERNEL_TRACE_PLAN=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp2_dispatch_ablation_tp2_8b"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp2_tp2.log"

{
    echo "=== exp2 TP=2 8B dispatch-space ablation started $(date -Iseconds) ==="
    echo "cell:  L_p=8192, K=32, B=8, max_new=2048"
    echo "model: $BE_MODEL  (tp=2)"
    echo "GPUs:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
    echo
} | tee "$LOG"

methods=(bs_kernel bs_kernel_2l1p bs_kernel_3l1p bs_kernel_4l1p bs_kernel_5l1p bs_kernel_2ldt bs_kernel_3ldt bs_kernel_4ldt bs_kernel_5ldt)
for m in "${methods[@]}"; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_batched.py \
        --K 32 --L_p 8192 --B 8 --max_new 2048 \
        --prompts-file data/gov_report.jsonl \
        --max_pages 40000 \
        --warmup \
        --measure_start 1500 --measure_end 2000 \
        --methods "$m" \
        --out "$OUT_DIR/${m}.csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

MERGED="$OUT_DIR/merged.csv"
echo "=== merging CSVs to $MERGED ===" | tee -a "$LOG"
first=1
for m in "${methods[@]}"; do
    f="$OUT_DIR/${m}.csv"
    [[ -f "$f" ]] || { echo "WARN: $f missing" | tee -a "$LOG"; continue; }
    if [[ $first -eq 1 ]]; then
        cat "$f" > "$MERGED"; first=0
    else
        tail -n +2 "$f" >> "$MERGED"
    fi
done

{
    echo
    echo "=== merged results ==="
    cat "$MERGED"
    echo
    echo "Done. Results: $OUT_DIR"
} | tee -a "$LOG"

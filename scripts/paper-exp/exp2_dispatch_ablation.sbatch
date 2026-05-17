#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/paper_exp2_%j.out
#SBATCH --error=slurm/logs/paper_exp2_%j.err

# Reproduces docs/paper_story.md → "Dispatch-space ablation" subsection.
#
# Forces each candidate strategy on the long-decode beam-search cell
# (Llama-3.2-1B, K=32, L_p=8192, B=16, max_new=2048) and compares
# against the default picker. Strategy wrappers are defined in
# benchmarks/bs_kernel/bench_batched.py:65-150 via
# `available_strategies={Strategy.SHARED_*L_*}`.
#
# Variants (depth-by-depth, both FUSED and DECTAIL families):
#   default           — bs_kernel       (picker free to choose)
#   forced 2L_FUSED   — bs_kernel_2l1p   (cascade depth baseline)
#   forced 3L_FUSED   — bs_kernel_3l1p
#   forced 4L_FUSED   — bs_kernel_4l1p
#   forced 2L_DECTAIL — bs_kernel_2ldt   (decode-tail baseline)
#   forced 3L_DECTAIL — bs_kernel_3ldt
#   forced 4L_DECTAIL — bs_kernel_4ldt
# Comparing 2L → 3L → 4L within each family makes the marginal-benefit
# argument explicit: depth-3 gains over depth-2, depth-4 does not gain
# (and adds plan cost) — same shape for the DECTAIL family.
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp2_dispatch_ablation/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B
export BS_KERNEL_TRACE_PLAN=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp2_dispatch_ablation"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp2.log"

{
    echo "=== exp2 dispatch-space ablation started $(date -Iseconds) ==="
    echo "cell:  L_p=8192, K=32, B=16, max_new=2048"
    echo "model: $BE_MODEL"
    echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

methods=(bs_kernel bs_kernel_2l1p bs_kernel_3l1p bs_kernel_4l1p bs_kernel_2ldt bs_kernel_3ldt bs_kernel_4ldt)
for m in "${methods[@]}"; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 32 --L_p 8192 --B 16 --max_new 2048 \
        --prompts-file data/gov_report.jsonl \
        --max_pages 80000 \
        --warmup \
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

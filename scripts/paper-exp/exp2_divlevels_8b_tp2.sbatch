#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_exp2_div8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=01:40:00
#SBATCH --output=slurm/logs/exp2_divlevels_8b_%j.out
#SBATCH --error=slurm/logs/exp2_divlevels_8b_%j.err

# Redone TP=2 Llama-3.1-8B dispatch-space depth ablation. Supersedes
# exp2_tp2_8b.sbatch, whose deeper DECTAIL rows silently fell back to 2L
# (the cost model picked the cheaper shallow variant from the alias's
# fallback list, so 4L/5L_DECTAIL = 2L + noise).
#
# Two changes make every depth a genuine measurement:
#   1. Deeper cascade levels come from BEAM-SEARCH DIVERGENCE, gated so a
#      divergence only becomes its own shared level when it shares
#      > 128 tokens across > 8 beams (BS_KERNEL_MIN_LEVEL_TOKENS / _CHILDREN).
#   2. Each forced 2L/3L/4L/5L variant is HARD-forced to its single
#      strategy family; the cost model only auto-falls-back to a shallower
#      depth when the beam tree genuinely lacks that many qualifying levels
#      (reported in the BS_KERNEL_TRACE_PLAN per-method strategy mix).
#
# Cell: gov_report long-decode beam search, K=32 L_p=8192 B=8 max_new=2048,
#       forward measured over decode steps 1500-2000. B=8 (not 16) so the
#       page-table eager allocation fits 80 GiB/rank; matches 70B/TP=2.
# Output: benchmarks/bs_kernel/results/paper-exp/exp2_divlevels_8b_tp2/

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
export BE_MODEL=meta-llama/Llama-3.1-8B
export BS_KERNEL_TRACE_PLAN=1
# Divergence-level gate set to its NATURAL MINIMUM (a fork becomes a shared
# level when >=2 beams share >=1 page). The 1B sweep (exp2_thr_sweep_1b)
# showed gov_report beam divergence is fine-grained — built from 2-beam
# clusters — so any children>2 gate kills 100% of deep levels and a token
# gate steadily erodes them (5L fires 87% at >1/>0, 32% at >1/>16, 0% at
# >2/*). The natural minimum is the only setting where each forced depth
# actually runs at that depth for most steps, giving an interpretable
# per-depth forward-time table.
export BS_KERNEL_MIN_LEVEL_TOKENS=0
export BS_KERNEL_MIN_LEVEL_CHILDREN=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp2_divlevels_8b_tp2"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp2_divlevels_8b.log"

{
    echo "=== exp2 divergence-level ablation TP=2 8B $(date -Iseconds) ==="
    echo "cell:  L_p=8192, K=32, B=8, max_new=2048, window=[1500,2000)"
    echo "model: $BE_MODEL  (tp=2)"
    echo "gate:  MIN_LEVEL_TOKENS=$BS_KERNEL_MIN_LEVEL_TOKENS MIN_LEVEL_CHILDREN=$BS_KERNEL_MIN_LEVEL_CHILDREN"
    echo "git:   $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
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

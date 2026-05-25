#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp2_div1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/exp2_divlevels_1b_%j.out
#SBATCH --error=slurm/logs/exp2_divlevels_1b_%j.err

# DIAGNOSTIC (cheap, 1B single-GPU) for the redone dispatch-space depth
# ablation. The deeper cascade levels in this experiment come from BEAM
# SEARCH DIVERGENCE, gated so a divergence only becomes its own shared
# level when it shares > 128 tokens across > 8 beams:
#     BS_KERNEL_MIN_LEVEL_TOKENS=128
#     BS_KERNEL_MIN_LEVEL_CHILDREN=8
# Each forced 2L/3L/4L/5L (FUSED=1POOL, DECTAIL) variant is HARD-forced to
# its single strategy family (cost model auto-falls-back to the deepest
# supported depth only when the beam tree lacks that many qualifying
# levels). BS_KERNEL_TRACE_PLAN=1 prints the per-method strategy mix =
# the realized per-step depth histogram, which is what we inspect here
# before spending the 8B/TP=2 run.
#
# Cell: gov_report long-decode beam search, K=32 L_p=8192 B=8 max_new=2048,
#       forward measured over decode steps 1500-2000.
# Output: benchmarks/bs_kernel/results/paper-exp/exp2_divlevels_1b/

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
export BE_MODEL=meta-llama/Llama-3.2-1B-Instruct   # cached in /work; gated base 1B is not
export BE_DTYPE=bf16            # 1B has head_dim=64; FA3/SM90 build bug → FA2
export BE_KV_DTYPE=bf16
export BE_FORCE_FA2=1
export BS_KERNEL_TRACE_PLAN=1
export BS_KERNEL_MIN_LEVEL_TOKENS=32
export BS_KERNEL_MIN_LEVEL_CHILDREN=2

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp2_divlevels_1b"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp2_divlevels_1b.log"

{
    echo "=== exp2 divergence-level ablation (1B diagnostic) $(date -Iseconds) ==="
    echo "cell:  L_p=8192, K=32, B=8, max_new=2048, window=[1500,2000)"
    echo "model: $BE_MODEL  (tp=1, FA2)"
    echo "gate:  MIN_LEVEL_TOKENS=$BS_KERNEL_MIN_LEVEL_TOKENS MIN_LEVEL_CHILDREN=$BS_KERNEL_MIN_LEVEL_CHILDREN"
    echo "git:   $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
    echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

methods=(bs_kernel bs_kernel_2l1p bs_kernel_3l1p bs_kernel_4l1p bs_kernel_5l1p bs_kernel_2ldt bs_kernel_3ldt bs_kernel_4ldt bs_kernel_5ldt)
for m in "${methods[@]}"; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_batched.py \
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

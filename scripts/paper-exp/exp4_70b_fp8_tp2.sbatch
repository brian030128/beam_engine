#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp4_70b_fp8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/paper_exp4_70b_fp8_%j.out
#SBATCH --error=slurm/logs/paper_exp4_70b_fp8_%j.err

# TP=2 long-decode beam search on RedHatAI/Meta-Llama-3-70B-Instruct-FP8.
# Sister of exp4_tp2_8b.sbatch with two cell-shape changes to fit
# H100 TP=2 at fp8 weight footprint:
#   - max_new lowered 2048 → 768  (per-step ~8x heavier on 70B)
#   - B halved 8 → 4              (35.81 GiB fp8 weights + decode-page
#                                   KV at max_new=768 needs the lower B)
# fasttree and mlca run alongside paged and bs_kernel to cover the
# same baseline matrix as the 8B variant.
#
# Compute dtype bf16 (Llama-3 base); KV bf16 (fp8 KV deferred).
# Ephemeral HF cache so the 70 GB download doesn't pollute /work.
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp4_70b_fp8_tp2/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp4_70b_fp8_tp2"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp4_70b_fp8.log"

{
    echo "=== exp4 TP=2 70B-FP8 long-decode beam search started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=bf16)"
    echo "prompts:  data/gov_report.jsonl"
    echo "cell:     L_p=8192, K=32, B=4, max_new=768"
    echo "methods:  paged fasttree mlca bs_kernel"
    echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
    echo
} | tee "$LOG"

methods=(paged fasttree mlca bs_kernel)
for m in "${methods[@]}"; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    trace_env=""
    if [[ "$m" == "bs_kernel" ]]; then
        trace_env="BS_KERNEL_TRACE_PLAN=1"
    fi
    env $trace_env torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_batched.py \
        --K 32 --L_p 8192 --B 4 --max_new 768 \
        --prompts-file data/gov_report.jsonl \
        --warmup \
        --methods "$m" \
        --out "$OUT_DIR/${m}.csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (method $m failed; continuing)" | tee -a "$LOG"
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
    [[ -f "$MERGED" ]] && cat "$MERGED" || echo "(no CSVs produced)"
    echo
    echo "Done. Results: $OUT_DIR"
} | tee -a "$LOG"

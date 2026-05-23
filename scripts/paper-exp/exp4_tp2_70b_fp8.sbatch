#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp4_70b_fp8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/paper_exp4_70b_fp8_%j.out
#SBATCH --error=slurm/logs/paper_exp4_70b_fp8_%j.err

# TP=2 Llama-3-70B-Instruct-FP8 long-decode beam search.
#
# Cell is scaled down vs exp4_tp2_8b (K=32 L_p=8192 B=8 max_new=2048).
# Per-rank weight residency is 35.81 GiB for the fp8 70B (vs ~8 GiB
# for the 8B bf16 baseline), leaving ~40 GiB for KV+activations. The
# original exp4 cell would need ~120 GiB of KV per rank at 70B; this
# adapted cell (K=16, L_p=8192, B=4, max_new=512) fits comfortably and
# keeps the long-decode shape (long prefix shared across beams + tail
# growth) that exp4 is designed to measure.
#
# Methods: paged, mlca, bs_kernel, fasttree.
# (tree's per-prompt fused buffer + custom tree-mask kernel would
# also work since dtype == kv_dtype, but is omitted to keep wall-time
# under the 60-min dev cap.)
#
# kv_dtype defaults to compute dtype (bf16). Native fp8 KV is deferred
# per project_fp8_kv_cache_blocker (FlashInfer tvm_ffi can't dispatch
# fp8 / uint8 through append_paged_kv_cache yet).
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp4_long_decode_tp2_70b_fp8/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# Node-local ephemeral HF cache — same recipe as the 70B smoke; the
# fp8 70B checkpoint (~70 GB) must NOT persist into /work/u4320956.
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE="bf16"  # bf16 compute → matches Llama-3-70B's training dtype.

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp4_long_decode_tp2_70b_fp8"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp4_70b_fp8.log"

{
    echo "=== exp4 TP=2 70B-fp8 long-decode beam search started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2)"
    echo "dtype:    $BE_DTYPE  (kv_dtype=bf16; fp8 KV deferred)"
    echo "prompts:  data/gov_report.jsonl"
    echo "cell:     L_p=8192, K=16, B=4, max_new=512"
    echo "methods:  paged mlca bs_kernel fasttree"
    echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
    echo
} | tee "$LOG"

methods=(paged mlca bs_kernel fasttree)
for m in "${methods[@]}"; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    trace_env=""
    if [[ "$m" == "bs_kernel" ]]; then
        trace_env="BS_KERNEL_TRACE_PLAN=1"
    fi
    env $trace_env torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_batched.py \
        --K 16 --L_p 8192 --B 4 --max_new 512 \
        --prompts-file data/gov_report.jsonl \
        --warmup \
        --methods "$m" \
        --out "$OUT_DIR/${m}.csv" \
        2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "  >>> ${m}: torchrun exited with $rc" | tee -a "$LOG"
    fi
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
    cat "$MERGED" 2>/dev/null || echo "(no merged CSV — all methods failed)"
    echo
    echo "Done. Results: $OUT_DIR"
} | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp4_70b_fp8_K128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/paper_exp4_70b_fp8_K128_%j.out
#SBATCH --error=slurm/logs/paper_exp4_70b_fp8_K128_%j.err

# K=128 variant of exp4_70b_fp8_tp2. Aim: probe whether bs_kernel's
# margin over paged grows when K scales (the dispatch space's
# advantage scales with K — more beams ⇒ more prefix sharing for the
# cascade, more sensitivity to the per-beam tail's CTA_Q mismatch).
#
# Cell: K=128, L_p=8192, B=2, max_new=768 (B halved 4→2 so the
# per-rank KV slab fits at K=128; total beams 256 matches the K=32/B=8
# comparison cell). bs_kernel's win at K=32/B=4 was only 1.03× over
# paged — if the K-sensitivity hypothesis is right, K=128 should
# widen that to 1.3-1.6×.
#
# Methods: paged, mlca, bs_kernel. fasttree skipped — at K=32 it was
# already 2.77× behind on this shape; at K=128 it would dominate the
# 60-min cap before producing useful comparison data.
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp4_70b_fp8_tp2_K128/

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

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp4_70b_fp8_tp2_K128"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp4_70b_fp8_K128.log"

{
    echo "=== exp4 TP=2 70B-FP8 long-decode K=128 started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=bf16)"
    echo "prompts:  data/gov_report.jsonl"
    echo "cell:     L_p=8192, K=128, B=2, max_new=768"
    echo "methods:  paged mlca bs_kernel"
    echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
    echo
} | tee "$LOG"

methods=(paged mlca bs_kernel)
for m in "${methods[@]}"; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    trace_env=""
    if [[ "$m" == "bs_kernel" ]]; then
        trace_env="BS_KERNEL_TRACE_PLAN=1"
    fi
    env $trace_env torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_batched.py \
        --K 128 --L_p 8192 --B 2 --max_new 768 \
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

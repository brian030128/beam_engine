#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp2_70b_fp8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/paper_exp2_70b_fp8_%j.out
#SBATCH --error=slurm/logs/paper_exp2_70b_fp8_%j.err

# TP=2 dispatch-space ablation on RedHatAI/Meta-Llama-3-70B-Instruct-FP8.
# Sister of exp2_tp2_8b.sbatch but on the FP8-quantized 70B
# checkpoint. Compute dtype is bf16 (the original training dtype for
# Llama-3-70B; fp16 would risk attention-score overflow at L_p=8k).
# KV cache stays bf16 — fp8 KV is deferred until FlashInfer's
# append_paged_kv_cache dispatches fp8/uint8 cleanly (see the project
# memory entry on the dispatch blocker).
#
# Scope vs the 8B version:
#   - max_new lowered 2048 → 256  (per-step ~8x heavier on 70B; we
#     need wall-clock + KV footprint to fit two H100s on a 60-min cap)
#   - methods: full 7-variant ablation matching exp2_tp2_8b (free picker
#     + 6 forced strategies: 2L_1POOL, 2L_DT, 3L_1POOL, 3L_DT, 4L_1POOL,
#     4L_DT). At ~30 s wall per method on this cell, 7 methods + 7 loads
#     fits within the 60-min cap.
#   - B halved 16 → 8 (~36 GiB fp8 weights + ~25 GiB bf16 KV per rank
#     at B=16/K=32/L_p=8192/max_new=256 = ~61 GiB, too tight at TP=2;
#     B=8 sits comfortably at ~48 GiB/rank)
#   - --max_pages dropped: the 8B sbatch's 80000 was sized for the 8B
#     (32-layer) cache; on 70B (80 layers) it would allocate ~196 GiB
#     of KV pages. Let bench_batched.run_one auto-size from B/K/L_p.
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp2_70b_fp8_tp2/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# Ephemeral node-local HF cache — never persists into /work/u4320956.
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BS_KERNEL_TRACE_PLAN=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp2_70b_fp8_tp2"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp2_70b_fp8.log"

{
    echo "=== exp2 TP=2 70B-FP8 dispatch-space ablation started $(date -Iseconds) ==="
    echo "cell:  L_p=8192, K=32, B=8, max_new=256"
    echo "model: $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=bf16)"
    echo "GPUs:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
    echo
} | tee "$LOG"

methods=(bs_kernel bs_kernel_2l1p bs_kernel_3l1p bs_kernel_4l1p bs_kernel_2ldt bs_kernel_3ldt bs_kernel_4ldt)
for m in "${methods[@]}"; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_batched.py \
        --K 32 --L_p 8192 --B 8 --max_new 256 \
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

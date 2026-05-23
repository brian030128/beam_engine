#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp4_70b_fp8_K128_slow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/paper_exp4_70b_fp8_K128_slow_%j.out
#SBATCH --error=slurm/logs/paper_exp4_70b_fp8_K128_slow_%j.err

# Companion to exp4_70b_fp8_tp2_K128.sbatch — fasttree + deft, the two
# slower methods on this shape that were omitted from the primary
# K=128 run to keep the wall time short.
#
# Estimated wall: fasttree per-step ~640 ms (4x K=32 baseline of 160
# ms), 767 steps × 2 (warmup+timed) ≈ 16 min. DeFT per-step at 70B
# K=128 is the wildcard — at 1B/K=32 it was ~150 ms/step (20-30x
# bs_kernel); 70B scales forward ~8x, K=128 scales tail ~4x. Rough
# estimate: 5-30 min for DeFT. Both should fit within 59 min.
#
# Cell: K=128, L_p=8192, B=2, max_new=768 (matches the primary K=128
# sbatch so CSVs merge into the same comparison).
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp4_70b_fp8_tp2_K128_slow/

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

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp4_70b_fp8_tp2_K128_slow"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp4_70b_fp8_K128_slow.log"

{
    echo "=== exp4 TP=2 70B-FP8 K=128 slow methods (fasttree, deft) ==="
    echo "started $(date -Iseconds)"
    echo "model:    $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=bf16)"
    echo "cell:     L_p=8192, K=128, B=2, max_new=768"
    echo "methods:  fasttree deft"
    echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
    echo
} | tee "$LOG"

# FastTree first — well-bounded.
echo "=== running method: fasttree ===" | tee -a "$LOG"
torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/bench_batched.py \
    --K 128 --L_p 8192 --B 2 --max_new 768 \
    --prompts-file data/gov_report.jsonl \
    --warmup \
    --methods fasttree \
    --out "$OUT_DIR/fasttree.csv" \
    2>&1 | tee -a "$LOG" \
    || echo "  (fasttree failed; continuing)" | tee -a "$LOG"
echo | tee -a "$LOG"

# DeFT second — may be slow; skip warmup to free wall time and still
# get an honest comparison.
echo "=== running method: deft ===" | tee -a "$LOG"
torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/bench_batched.py \
    --K 128 --L_p 8192 --B 2 --max_new 768 \
    --prompts-file data/gov_report.jsonl \
    --methods deft \
    --out "$OUT_DIR/deft.csv" \
    2>&1 | tee -a "$LOG" \
    || echo "  (deft failed; continuing)" | tee -a "$LOG"
echo | tee -a "$LOG"

MERGED="$OUT_DIR/merged.csv"
echo "=== merging CSVs to $MERGED ===" | tee -a "$LOG"
first=1
for m in fasttree deft; do
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

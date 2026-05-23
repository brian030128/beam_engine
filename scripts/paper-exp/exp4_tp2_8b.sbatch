#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp4_tp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/paper_exp4_tp2_%j.out
#SBATCH --error=slurm/logs/paper_exp4_tp2_%j.err

# TP=2 Llama-3.1-8B long-decode beam search (companion to exp4).
# Cell: L_p=8192, K=32, B=8, max_new=2048. 1B exp4 ran B=16 but 8B at
# TP=2 still doesn't fit B=16 (per-rank KV slab ~75 GB > free budget);
# halving B halves the slab while keeping the long-decode beam-search
# shape (K=32, max_new=2048, L_p=8192) intact.
#
# Methods: paged, fasttree, mlca, bs_kernel (deft would go in its own
# follow-up sbatch since the vendored kernel is ~20-30x slower).
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp4_long_decode_tp2_8b/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.1-8B

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp4_long_decode_tp2_8b"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp4_tp2.log"

{
    echo "=== exp4 TP=2 8B long-decode beam search started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2)"
    echo "prompts:  data/gov_report.jsonl"
    echo "cell:     L_p=8192, K=32, B=8, max_new=2048"
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
        --K 32 --L_p 8192 --B 8 --max_new 2048 \
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
    cat "$MERGED"
    echo
    echo "Done. Results: $OUT_DIR"
} | tee -a "$LOG"

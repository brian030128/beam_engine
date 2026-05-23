#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_exp4_q3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/paper_exp4_qwen3_%j.out
#SBATCH --error=slurm/logs/paper_exp4_qwen3_%j.err

# Qwen3-4B long-decode beam search — companion to exp4_long_decode.sbatch.
# Same cell shape (K=32, L_p=8192, B=16, max_new=2048).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL="Qwen/Qwen3-4B"

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp4_long_decode_qwen3_4b"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp4_qwen3.log"

{
    echo "=== exp4 Qwen3-4B long-decode started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL"
    echo "HF_HOME:  $HF_HOME"
    echo "prompts:  data/gov_report.jsonl"
    echo "cell:     L_p=8192, K=32, B=16, max_new=2048"
    echo "methods:  paged fasttree mlca bs_kernel  (deft in exp4_deft_qwen3_4b.sbatch)"
    echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

methods=(paged fasttree mlca bs_kernel)
for m in "${methods[@]}"; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    trace_env=""
    if [[ "$m" == "bs_kernel" ]]; then
        trace_env="BS_KERNEL_TRACE_PLAN=1"
    fi
    env $trace_env uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 32 --L_p 8192 --B 4 --max_new 2048 \
        --prompts-file data/gov_report.jsonl \
        --max_pages 20000 \
        --warmup \
        --methods "$m" \
        --out "$OUT_DIR/${m}.csv" \
        2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "  >>> ${m}: python exited with $rc" | tee -a "$LOG"
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

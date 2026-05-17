#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/paper_exp4_%j.out
#SBATCH --error=slurm/logs/paper_exp4_%j.err

# Reproduces docs/paper_story.md → "Workload coverage: beam search at
# long decode length" subsection.
#
# Long-decode beam search at max_new=2048 (4.0× the paper grid).
#   model    : Llama-3.2-1B
#   cell     : L_p=8192, K=32, B=16, max_new=2048
#              B halved 32→16 because the worst-case KV slab for
#              B=32/K=32/max_new=2048 ~77 GB doesn't fit on a single
#              H100. 8B at this cell exceeds 80 GB so we don't try.
#   methods  : paged, fasttree, mlca, bs_kernel
#              (deft is in exp4_deft.sbatch — separated because the
#              vendored DeFT kernel is ~20-30x slower per step than the
#              others and would push the job over the 40-min wallclock
#              cap. Both sbatches write into the same OUT_DIR; the
#              deft variant appends to merged.csv.)
#   prompts  : data/gov_report.jsonl (distinct per-prompt prefixes)
#
# bs_kernel strategy picks logged via BS_KERNEL_TRACE_PLAN=1.
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp4_long_decode/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp4_long_decode"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp4.log"

{
    echo "=== exp4 long-decode beam search started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL"
    echo "prompts:  data/gov_report.jsonl"
    echo "cell:     L_p=8192, K=32, B=16, max_new=2048"
    echo "methods:  paged fasttree mlca bs_kernel  (deft in exp4_deft.sbatch)"
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
        --K 32 --L_p 8192 --B 16 --max_new 2048 \
        --prompts-file data/gov_report.jsonl \
        --max_pages 80000 \
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

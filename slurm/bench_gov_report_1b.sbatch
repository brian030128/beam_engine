#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=gov_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/bench_gov_report_1b_%j.out
#SBATCH --error=slurm/logs/bench_gov_report_1b_%j.err

# Beam-search + Diverse-Beam-Search on real gov_report prompts.
#   model      : Llama-3.2-1B
#   prompts    : data/gov_report.jsonl (200 docs, ≥ 8192 tokens each)
#   cell       : L_p=8192, K=32, B=32, max_new=256
#   methods    : paged, fasttree, mlca, bs_kernel  (× standard + DBS)
# Prompts are pairwise distinct → no cross-prompt prefix sharing.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/gov_report_1b-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== gov_report_1b started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL"
    echo "prompts:  data/gov_report.jsonl"
    echo "cell:     L_p=8192, K=32, B=32, max_new=256"
    echo "methods:  paged fasttree mlca bs_kernel + DBS variants"
    echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

methods=(paged fasttree mlca bs_kernel dbs_paged dbs_fasttree dbs_mlca dbs_bs_kernel)
for m in "${methods[@]}"; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 32 --L_p 8192 --B 32 --max_new 256 \
        --prompts-file data/gov_report.jsonl \
        --max_pages 40000 \
        --warmup \
        --methods "$m" \
        --out "$RESULTS_DIR/${m}.csv" \
        2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "  >>> ${m}: python exited with $rc" | tee -a "$LOG"
    fi
    echo | tee -a "$LOG"
done

MERGED="$RESULTS_DIR/merged.csv"
echo "=== merging CSVs to $MERGED ===" | tee -a "$LOG"
first=1
for m in "${methods[@]}"; do
    f="$RESULTS_DIR/${m}.csv"
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
    echo "Done. Results: $RESULTS_DIR"
} | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=gov_1b_lng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_gov_report_1b_long_%j.out
#SBATCH --error=slurm/logs/bench_gov_report_1b_long_%j.err

# Long-decode beam-search + DBS on gov_report.
#   model      : Llama-3.2-1B
#   prompts    : data/gov_report.jsonl (pairwise distinct, L_p tokens each)
#   cell       : L_p=8192, K=32, B=16, max_new=2048
#                (B halved from 32 → 16: at max_new=2048 the worst-case KV
#                 slab for B=32,K=32 is ~77 GB and doesn't fit alongside
#                 model + activations on a single 80 GB H100. B=16 → ~38 GB)
#   methods    : paged, fasttree, mlca, bs_kernel  (× standard + DBS)
# Records bs_kernel's per-step strategy picks via BS_KERNEL_TRACE_PLAN=1
# on its two invocations (standard + DBS).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/gov_report_1b_long-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== gov_report_1b_long started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL"
    echo "prompts:  data/gov_report.jsonl"
    echo "cell:     L_p=8192, K=32, B=16, max_new=2048"
    echo "methods:  paged fasttree mlca bs_kernel + DBS variants"
    echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

methods=(paged fasttree mlca bs_kernel dbs_paged dbs_fasttree dbs_mlca dbs_bs_kernel)
for m in "${methods[@]}"; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    # Enable plan trace only for the two bs_kernel invocations so the
    # report can show strategy picks per step.
    trace_env=""
    if [[ "$m" == "bs_kernel" || "$m" == "dbs_bs_kernel" ]]; then
        trace_env="BS_KERNEL_TRACE_PLAN=1"
    fi
    env $trace_env uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 32 --L_p 8192 --B 16 --max_new 2048 \
        --prompts-file data/gov_report.jsonl \
        --max_pages 80000 \
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

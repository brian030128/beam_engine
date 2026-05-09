#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_b32_k64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_b32_k64_lp8k_%j.out
#SBATCH --error=slurm/logs/bench_b32_k64_lp8k_%j.err

# Beam search benchmark at L_p=8192, K=64, B=32 across 5 methods:
# paged_attention, fasttree, mlca, adaptive_pool, bs_kernel.
#
# Each method runs in its own python invocation so KV-page memory from one
# method cannot starve the next (the previous combined-process B=32 run OOM'd
# at ~76 GiB resident — that's fragmentation accumulation, not true demand).

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_b32_k64_lp8k-${TS}"
mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/bench.log"
MERGED="$RESULTS_DIR/bench_b32_k64_lp8k.csv"

methods=(paged fasttree mlca adaptive_pool bs_kernel)

{
    echo "=== bench_b32_k64_lp8k started $(date -Iseconds) ==="
    echo "config: K=64, L_p=8192, B=32, max_new=16"
    echo "methods: ${methods[*]}"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

for m in "${methods[@]}"; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 64 \
        --L_p 8192 \
        --B 32 \
        --max_new 16 \
        --methods "$m" \
        --out "$RESULTS_DIR/${m}.csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

# Merge per-method CSVs (keep header from first, drop from rest).
echo "=== merging CSVs to $MERGED ===" | tee -a "$LOG"
first=1
for m in "${methods[@]}"; do
    f="$RESULTS_DIR/${m}.csv"
    if [[ ! -f "$f" ]]; then
        echo "WARN: $f missing, skipping" | tee -a "$LOG"
        continue
    fi
    if [[ $first -eq 1 ]]; then
        cat "$f" > "$MERGED"
        first=0
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

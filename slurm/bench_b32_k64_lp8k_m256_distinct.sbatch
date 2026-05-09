#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_d256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_d256_%j.out
#SBATCH --error=slurm/logs/bench_d256_%j.err

# K=64, L_p=8192, B=32, max_new=256, with B pairwise-distinct prompts.
#
# Each method runs in its own python invocation (memory isolation).
# bs_kernel pick histogram is dumped at the end via dump_picks_batched.

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_b32_k64_lp8k_m256_distinct-${TS}"
mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/bench.log"
MERGED="$RESULTS_DIR/bench.csv"

methods=(paged fasttree mlca adaptive_pool bs_kernel)

{
    echo "=== bench_b32_k64_lp8k_m256_distinct started $(date -Iseconds) ==="
    echo "config: K=64, L_p=8192, B=32, max_new=256, distinct=True"
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
        --max_new 256 \
        --distinct \
        --methods "$m" \
        --out "$RESULTS_DIR/${m}.csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

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

echo | tee -a "$LOG"
echo "=== merged results ===" | tee -a "$LOG"
cat "$MERGED" | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== bs_kernel pick histogram (max_new=256, distinct prompts) ===" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/dump_picks_batched.py \
    --K 64 --L_p 8192 --B 32 --max_new 256 --distinct \
    2>&1 | tee -a "$LOG" | tee "$RESULTS_DIR/picks.log"

echo | tee -a "$LOG"
echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bs_8b_4k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:35:00
#SBATCH --output=slurm/logs/bench_bs_8b_lp4k_%j.out
#SBATCH --error=slurm/logs/bench_bs_8b_lp4k_%j.err

# E2E beam-search bench for Llama-3.1-8B:
#   L_p=4096, K=32, B=32, max_new=256, all 4 methods
#   (fasttree, bs_kernel, paged, mlca), --warmup (single measured pass).
#
# Memory budget at this cell with default repeated prompts (paged
# refcounts shared prefix):
#   shared prefill = 256 pages (4096/16, shared via refcount)
#   decode = B*K*(256/16+2) = 1024*18 = 18432 pages
#   + 512 headroom → ~19200 pages
# At 8B, per-page slab = 2 MB → ~38 GB KV, fits with 16 GB model.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.1-8B

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bs_8b_lp4k-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== bs_8b_lp4k started $(date -Iseconds) ==="
    echo "config:  L_p=4096, K=32, B=32, max_new=256"
    echo "methods: fasttree bs_kernel paged mlca"
    echo "--warmup (1 discarded + 1 measured per method)"
    echo "model:   meta-llama/Llama-3.1-8B"
    echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

# Each method runs in its own python process — same pattern as
# slurm/bench_b32_k64_lp8k.sbatch — so a method's KV slab can't starve
# the next (and any OOM is isolated).
methods=(fasttree bs_kernel paged mlca)
for m in "${methods[@]}"; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 16 \
        --L_p 4096 \
        --B 32 \
        --max_new 256 \
        --max_pages 10500 \
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

# Merge per-method CSVs.
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

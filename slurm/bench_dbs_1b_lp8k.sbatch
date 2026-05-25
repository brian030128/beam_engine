#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=dbs_1b_8k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:35:00
#SBATCH --output=slurm/logs/bench_dbs_1b_lp8k_%j.out
#SBATCH --error=slurm/logs/bench_dbs_1b_lp8k_%j.err

# Diverse Beam Search (Vijayakumar et al. 2016) on the same cell as
# the regular-beam-search 1B sweep so the rank can be cross-compared:
#   Llama-3.2-1B, L_p=8192, K=32, B=32, max_new=256.
# DBS uses Hamming-diversity-penalised top-K (num_groups=4 default,
# lambda=0.5 default) — same attention kernels, different top-K selector.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/dbs_1b_lp8k-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== dbs_1b_lp8k started $(date -Iseconds) ==="
    echo "config:  L_p=8192, K=32, B=32, max_new=256"
    echo "methods: dbs_fasttree dbs_bs_kernel dbs_paged dbs_mlca"
    echo "--warmup (1 discarded + 1 measured per method)"
    echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

methods=(dbs_fasttree dbs_bs_kernel dbs_paged dbs_mlca)
for m in "${methods[@]}"; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 32 --L_p 8192 --B 32 --max_new 256 \
        --max_pages 36500 \
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

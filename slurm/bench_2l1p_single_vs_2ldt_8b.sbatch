#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=2l1ps_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/bench_2l1p_single_vs_2ldt_8b_%j.out
#SBATCH --error=slurm/logs/bench_2l1p_single_vs_2ldt_8b_%j.err

# Forced 2l1p (single-CTA_Q launch) vs 2ldt sweep on Llama-3.1-8B.
# Grid: L_p=8192, max_new=256
#   (K,B) ∈ { (16,1), (16,4), (16,32), (32,1), (32,4), (64,1), (64,4) }
# B=32 dropped at K∈{32,64} to keep KV pool within 80 GB.
#   Per-page slab on 8B ≈ 2 MB → 50000 pages ≈ 100 GB.
#   B*K ≤ 512 → ≤ ~30 GB KV slab + 16 GB model + workspace fits H100.
# Plan: /home/u4320956/.claude/plans/polymorphic-sleeping-trinket.md

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.1-8B
export HF_HOME=/work/u4320956/hf

mkdir -p slurm/logs
TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/2l1p_single_vs_2ldt_8b-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== 2l1p_single vs 2ldt | 8B | started $(date -Iseconds) ==="
    echo "grid:   L_p=8192, max_new=256"
    echo "cells:  (K=16,B∈{1,4,32}) (K=32,B∈{1,4}) (K=64,B∈{1,4})"
    echo "model:  $BE_MODEL"
    echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "flashinfer diff (single_launch must be live):"
    git -C 3rdparty/flashinfer diff --stat 2>&1 | sed 's/^/  /'
    echo
} | tee "$LOG"

methods=(bs_kernel_2l1p_single bs_kernel_2ldt)
# (K, B) cells — keep B*K ≤ 512 on 8B so the KV slab fits in 80 GB.
cells=(
    "16 1"
    "16 4"
    "16 32"
    "32 1"
    "32 4"
    "64 1"
    "64 4"
)

for m in "${methods[@]}"; do
    for cell in "${cells[@]}"; do
        K=$(echo "$cell" | awk '{print $1}')
        B=$(echo "$cell" | awk '{print $2}')
        MAX_PAGES=$(( B * (8192 + K * 256) / 16 + 1024 ))
        tag="${m}_K${K}_B${B}"
        csv="$RESULTS_DIR/${tag}.csv"
        echo "=== ${tag} | max_pages=${MAX_PAGES} ===" | tee -a "$LOG"
        uv run python benchmarks/bs_kernel/bench_batched.py \
            --K "$K" \
            --L_p 8192 \
            --B "$B" \
            --max_new 256 \
            --max_pages "$MAX_PAGES" \
            --warmup \
            --methods "$m" \
            --out "$csv" \
            2>&1 | tee -a "$LOG"
        rc=${PIPESTATUS[0]}
        if [[ $rc -ne 0 ]]; then
            echo "  >>> ${tag}: python exited with $rc" | tee -a "$LOG"
        fi
        echo | tee -a "$LOG"
    done
done

MERGED="$RESULTS_DIR/merged.csv"
echo "=== merging CSVs to $MERGED ===" | tee -a "$LOG"
first=1
for f in "$RESULTS_DIR"/*.csv; do
    [[ "$f" == "$MERGED" ]] && continue
    [[ -f "$f" ]] || continue
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

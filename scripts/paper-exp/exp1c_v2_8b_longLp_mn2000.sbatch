#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_lp_mn2k_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/pe_lp_mn2k_8b_%j.out
#SBATCH --error=slurm/logs/pe_lp_mn2k_8b_%j.err

# Long-prefix small-batch probe at max_new=2000 (was 256). Tests whether
# the FUSED-vs-DECTAIL crossover the picker discovered at small B
# remains over long decode lengths, where the per-beam tail grows
# 8× larger and DECTAIL's bw-side cost rises proportionally.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="meta-llama/Llama-3.1-8B"

MODEL_TAG="Llama-3.1-8B"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1c_v2_longLp_mn2000/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== long-Lp max_new=2000 probe ${MODEL_TAG} started $(date -Iseconds) ==="
    echo "cells: K=16 L_p=50000 B ∈ {1, 2, 4} max_new=2000"
    echo
} | tee "$LOG"

METHODS="bs_kernel bs_kernel_2l1p bs_kernel_2ldt"

run_cell () {
    local B="$1"; shift
    local csv="$OUT_DIR/K16_Lp50k_B${B}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== cell: K=16  L_p=50000  B=${B}  max_new=2000 ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    local first=1
    for m in $METHODS; do
        echo "  --- method: $m ---" | tee -a "$LOG"
        local tmp="$OUT_DIR/K16_Lp50k_B${B}_${m}.csv"
        uv run python benchmarks/bs_kernel/bench_batched.py \
            --K 16 --L_p 50000 --B "$B" --max_new 2000 \
            --warmup --methods "$m" --out "$tmp" \
            2>&1 | tee -a "$LOG" \
            || { echo "  ($m failed; continuing)" | tee -a "$LOG"; continue; }
        if [[ $first -eq 1 ]]; then
            [[ -f "$tmp" ]] && cp "$tmp" "$csv" && first=0
        else
            [[ -f "$tmp" ]] && tail -n +2 "$tmp" >> "$csv"
        fi
    done
}

run_cell 1
run_cell 2
run_cell 4

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

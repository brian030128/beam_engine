#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_lp_mn2k_tp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/pe_lp_mn2k_tp2_%j.out
#SBATCH --error=slurm/logs/pe_lp_mn2k_tp2_%j.err

# TP=2 variant of the long-Lp K=16 L_p=50K probe at max_new=2000.
# Single-GPU B=4 OOM'd at mn=2000 (KV slab ~80GB). TP=2 halves
# per-rank num_kv_heads → fits comfortably.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export HF_HOME=/work/u4320956/hf
export HF_HUB_OFFLINE=1
export BE_MODEL="meta-llama/Llama-3.1-8B"

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1c_v2_longLp_mn2000_tp2/Llama-3.1-8B"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== long-Lp mn=2000 TP=2 ${BE_MODEL} started $(date -Iseconds) ==="
    echo "cells: K=16 L_p=50000 B ∈ {1, 2, 4} mn=2000 (TP=2)"
} | tee "$LOG"

METHODS="bs_kernel bs_kernel_2l1p bs_kernel_2ldt"

run_cell () {
    local B="$1"; shift
    local csv="$OUT_DIR/K16_Lp50k_B${B}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== cell: K=16 L_p=50000 B=${B} mn=2000 (TP=2) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    local first=1
    for m in $METHODS; do
        echo "  --- method: $m ---" | tee -a "$LOG"
        local tmp="$OUT_DIR/K16_Lp50k_B${B}_${m}.csv"
        torchrun --standalone --nproc_per_node=2 \
            benchmarks/bs_kernel/bench_batched.py \
            --K 16 --L_p 50000 --B "$B" --max_new 2000 \
            --warmup --methods "$m" --out "$tmp" \
            2>&1 | tee -a "$LOG" \
            || { echo "  ($m failed)" | tee -a "$LOG"; continue; }
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

echo "Done $(date -Iseconds)" | tee -a "$LOG"

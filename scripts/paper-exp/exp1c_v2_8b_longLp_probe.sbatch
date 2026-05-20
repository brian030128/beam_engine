#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_longLp_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/paper_longLp_8b_%j.out
#SBATCH --error=slurm/logs/paper_longLp_8b_%j.err

# Long-prefix small-batch probe — Llama-3.1-8B, K=16, L_p=50_000, B small.
#
# Hypothesis: at a 50K-token prefix the cascade's shared-prefix read
# dominates per-step cost, and the small per-branch tail (K=16, G=4 →
# R=4) is too narrow for the FUSED prefill template to amortize launch
# overhead, so SHARED_2L_DEC_TAIL should keep winning. At very small B
# the DEC_TAIL launch + merge overhead grows relative to the small
# tail-side work, so somewhere in B ∈ {1, 2, 4} we should see FUSED
# catch up or pass DEC_TAIL.
#
# Cells (bench_batched.py with --prompts-file pads to requested L_p):
#   K=16  L_p=50000  B=1  max_new=256
#   K=16  L_p=50000  B=2  max_new=256
#   K=16  L_p=50000  B=4  max_new=256
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp1c_v2_longLp/Llama-3.1-8B/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="meta-llama/Llama-3.1-8B"

MODEL_TAG="Llama-3.1-8B"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1c_v2_longLp/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== long-Lp probe ${MODEL_TAG} started $(date -Iseconds) ==="
    echo "model: $BE_MODEL  (tp=1)"
    echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "cells: K=16 L_p=50000 B ∈ {1, 2, 4} max_new=256"
    echo
} | tee "$LOG"

# bench_batched.py aliases: bs_kernel (picker), bs_kernel_2l1p (forced
# FUSED), bs_kernel_2ldt (forced DEC_TAIL). Run each separately so
# BS_KERNEL_TRACE_PLAN dumps fire cleanly per method.
METHODS="bs_kernel bs_kernel_2l1p bs_kernel_2ldt"

run_cell () {
    # run_cell <B>
    local B="$1"; shift
    local csv="$OUT_DIR/K16_Lp50k_B${B}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== cell: K=16  L_p=50000  B=${B}  max_new=256 ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    local first=1
    for m in $METHODS; do
        echo "  --- method: $m ---" | tee -a "$LOG"
        local tmp="$OUT_DIR/K16_Lp50k_B${B}_${m}.csv"
        # No --prompts-file: bench_batched.py synthesizes prompts via
        # _make_prompt to exactly L_p tokens. gov_report's longest
        # prompts are under 50K, so file mode skips every cell here.
        uv run python benchmarks/bs_kernel/bench_batched.py \
            --K 16 --L_p 50000 --B "$B" --max_new 256 \
            --warmup \
            --methods "$m" \
            --out "$tmp" \
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

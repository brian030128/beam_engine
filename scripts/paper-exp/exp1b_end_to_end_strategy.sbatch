#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/paper_exp1b_%j.out
#SBATCH --error=slurm/logs/paper_exp1b_%j.err

# Reproduces the strategy-mix column of the End-to-end speedups table.
#
# Runs bs_kernel only across the six paper cells with
# BS_KERNEL_TRACE_PLAN=1 so the picker's per-step pick histogram is
# printed inline. --repeat 1 (the trace adds host overhead so this is
# not a timing run — see exp1a_end_to_end_timing.sbatch for that).
#
# K=64 override across all cells (matches exp1a). At K=32 the picker
# chose SHARED_2L_1POOL 100% of steps; K=64 doubles the per-beam tail
# tile count, the regime where DEC_TAIL is meant to win.
#
# The strategy mix per cell is printed as 'strategy mix: {...}' near
# the end of each beam_search call.
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/trace/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/trace"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp1b.log"

{
    echo "=== exp1b strategy-mix trace started $(date -Iseconds) ==="
    echo "BS_KERNEL_TRACE_PLAN=1  --warmup --repeat 1"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    export BE_MODEL="meta-llama/${model_tag}"
    if [[ "$model_tag" == "Llama-3.2-1B" ]]; then
        CR_B=32; FS_B=32
    else
        CR_B=8;  FS_B=8
    fi
    echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"

    echo "  multi_level_system (B=4, K=64)" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_level_system --methods bs_kernel \
        --paper-exact --k-override 64 \
        --max-new 256 --repeat 1 --warmup \
        --out "$OUT_DIR/${model_tag}_mls.csv" \
        2>&1 | tee -a "$LOG"

    echo "  multi_few_shot (B=${FS_B}, K=64)" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_few_shot --methods bs_kernel \
        --paper-exact --k-override 64 --b-override $FS_B \
        --max-new 256 --repeat 1 --warmup \
        --out "$OUT_DIR/${model_tag}_mfs.csv" \
        2>&1 | tee -a "$LOG"

    echo "  multi_chain_reasoning (B=${CR_B}, K=64)" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_chain_reasoning --methods bs_kernel \
        --paper-exact --k-override 64 --b-override $CR_B \
        --max-new 256 --repeat 1 --warmup \
        --out "$OUT_DIR/${model_tag}_mcr.csv" \
        2>&1 | tee -a "$LOG"
done

echo "Done. Results: $OUT_DIR" | tee -a "$LOG"
echo "Strategy mix per cell is inline in $LOG (search for 'strategy mix:')" | tee -a "$LOG"

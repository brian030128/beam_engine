#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_exp1b_q3b16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/paper_exp1b_qwen3_b16_%j.out
#SBATCH --error=slurm/logs/paper_exp1b_qwen3_b16_%j.err

# Qwen3-4B strategy-mix trace — companion to exp1b_end_to_end_strategy.sbatch.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="Qwen/Qwen3-4B"

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/trace_b16"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp1b_qwen3_b16.log"

{
    echo "=== exp1b Qwen3-4B strategy-mix trace started $(date -Iseconds) ==="
    echo "model: $BE_MODEL"
    echo "HF_HOME: $HF_HOME"
    echo "BS_KERNEL_TRACE_PLAN=1  --warmup --repeat 1"
    echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

model_tag="Qwen3-4B"
CR_B=16
FS_B=16

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

echo "Done. Results: $OUT_DIR" | tee -a "$LOG"
echo "Strategy mix per cell is inline in $LOG (search for 'strategy mix:')" | tee -a "$LOG"

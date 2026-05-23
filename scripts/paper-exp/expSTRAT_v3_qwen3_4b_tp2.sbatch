#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_expSTRATv3_q3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_expSTRATv3_qwen3_4b_tp2_%j.out
#SBATCH --error=slurm/logs/paper_expSTRATv3_qwen3_4b_tp2_%j.err

# expSTRAT v3 — single cell at K=128 B=32 mn=128 (R=4096, mid_tail≈64).
# Same R as BR3 (K=64 B=64 mn=128) but with K doubled and B halved —
# tests whether per-prompt beam width matters at fixed R.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export HF_HOME=/work/u4320956/hf
export BE_MODEL="Qwen/Qwen3-4B"

MODEL_TAG="Qwen3-4B"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/expSTRATv3/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== expSTRATv3 ${MODEL_TAG} TP=2 started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2)"
    echo
} | tee "$LOG"

METHODS="paged bs_kernel bsk_2l_1p bsk_2l_dt"

K=128; B=32; mn=128; sc="multi_chain_reasoning"
csv="$OUT_DIR/${sc}_K${K}_B${B}_mn${mn}.csv"
{
    echo "=========================================================="
    echo "=== BR4: ${sc}  (K=${K}, B=${B}, max_new=${mn}, R=$((B*K)), tp=2) ==="
    echo "=========================================================="
} | tee -a "$LOG"

torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods $METHODS \
    --paper-exact --no-stage2 \
    --max-new "$mn" --repeat 1 --warmup \
    --scenarios "$sc" --k-override "$K" --b-override "$B" \
    --out "$csv" \
    2>&1 | tee -a "$LOG"

{
    echo
    echo "=== done $(date -Iseconds) ==="
} | tee -a "$LOG"

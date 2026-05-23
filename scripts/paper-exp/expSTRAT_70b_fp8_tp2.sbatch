#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_expSTRAT_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/paper_expSTRAT_70b_fp8_tp2_%j.out
#SBATCH --error=slurm/logs/paper_expSTRAT_70b_fp8_tp2_%j.err

# expSTRAT — strategy-separating cells for the paper,
# Llama-3-70B-FP8 TP=2 with fp8 weights AND fp8 KV
# (BE_KV_DTYPE=fp8_e4m3). FP8 KV halves the per-rank KV footprint, so
# the Family-B cells (K=64..128 × max_new=1024) fit on H100 80 GiB.
#
#   tag | scenario              | K   | B  | max_new | regime
#   ----+-----------------------+-----+----+---------+--------
#   A1  | multi_chain_reasoning |   2 | 32 |      32 | thin tail
#   A2  | multi_chain_reasoning |   4 | 16 |      32 | thin tail
#   A3  | multi_few_shot        |   4 |  8 |      32 | thin tail (long L_p)
#   B1  | multi_chain_reasoning |  64 |  4 |    1024 | thick tail
#   B2  | multi_chain_reasoning | 128 |  2 |    1024 | thick tail
#   B3  | multi_few_shot        |  64 |  4 |     512 | thick tail (long L_p)
#   C1  | multi_chain_reasoning |  32 | 16 |     256 | neutral
#   C2  | multi_level_system    |  32 |  4 |     256 | neutral
#
# Output: benchmarks/bs_kernel/results/paper-exp/expSTRAT/Llama-3-70B-FP8/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# 70B FP8 weights too large for /work — use ephemeral SLURM_TMPDIR.
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3
export BS_KERNEL_TRACE_PLAN=1

MODEL_TAG="Llama-3-70B-FP8"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/expSTRAT/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== expSTRAT ${MODEL_TAG} TP=2 started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo "GPUs:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) × 2"
    echo
} | tee "$LOG"

METHODS="paged bs_kernel bsk_2l_1p bsk_2l_dt"

run_cell () {
    local tag="$1"; shift
    local scenario="$1"; shift
    local K="$1"; shift
    local B="$1"; shift
    local mn="$1"; shift
    local csv="$OUT_DIR/${scenario}_K${K}_B${B}_mn${mn}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== ${tag}: ${scenario}  (K=${K}, B=${B}, max_new=${mn}, tp=2) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS \
        --paper-exact --no-stage2 \
        --max-new "$mn" --repeat 1 --warmup \
        --scenarios "$scenario" --k-override "$K" --b-override "$B" \
        --out "$csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (cell $tag failed; continuing)" | tee -a "$LOG"
}

# Family A — thin tail (2L_1POOL ≥30% expected)
run_cell A1 multi_chain_reasoning   2 32  32
run_cell A2 multi_chain_reasoning   4 16  32
run_cell A3 multi_few_shot          4  8  32

# Family B — thick tail (2L_DEC_TAIL ≥30% expected)
run_cell B1 multi_chain_reasoning  64  4 1024
run_cell B2 multi_chain_reasoning 128  2 1024
run_cell B3 multi_few_shot         64  4  512

# Family C — neutral (gap ≤5%)
run_cell C1 multi_chain_reasoning  32 16  256
run_cell C2 multi_level_system     32  4  256

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

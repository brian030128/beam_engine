#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_expSTRATv7_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_expSTRATv7_llama8b_tp1_%j.out
#SBATCH --error=slurm/logs/paper_expSTRATv7_llama8b_tp1_%j.err

# expSTRAT v7 — Family-A push on Llama-3.1-8B-FP8 TP=1.
# v6 LA4 (mcr K=2 B=4 mn=32, R=8) got 1P winning by 7%. Hypothesis:
# the (R × per-beam-tail-length) product still wasn't tiny enough for
# DT's per-step launch+merge overhead to dominate. mcr is the only
# scenario with L_priv=0, so tail = decoded tokens only — minimal.
#
#   tag | scenario              | K | B | max_new | R | mid-tail
#   ----+-----------------------+---+---+---------+---+---------
#   LA5 | multi_chain_reasoning | 2 | 2 |      32 | 4 | ≈16
#   LA6 | multi_chain_reasoning | 2 | 2 |       8 | 4 | ≈ 4  (super-short decode)
#   LA7 | multi_chain_reasoning | 2 | 1 |      32 | 2 | ≈16  (single prompt)
#   LA8 | multi_chain_reasoning | 4 | 1 |      32 | 4 | ≈16

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME=/work/u4320956/hf
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3

MODEL_TAG="Llama-3.1-8B-FP8"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/expSTRATv7/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== expSTRATv7 ${MODEL_TAG} TP=1 started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=1, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
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
        echo "=== ${tag}: ${scenario}  (K=${K}, B=${B}, max_new=${mn}, R=$((B*K)), tp=1, fp8) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=1 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS \
        --paper-exact --no-stage2 \
        --max-new "$mn" --repeat 1 --warmup \
        --scenarios "$scenario" --k-override "$K" --b-override "$B" \
        --out "$csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (cell $tag failed; continuing)" | tee -a "$LOG"
}

run_cell LA5 multi_chain_reasoning 2 2 32
run_cell LA6 multi_chain_reasoning 2 2  8
run_cell LA7 multi_chain_reasoning 2 1 32
run_cell LA8 multi_chain_reasoning 4 1 32

{
    echo
    echo "=== done $(date -Iseconds) ==="
} | tee -a "$LOG"

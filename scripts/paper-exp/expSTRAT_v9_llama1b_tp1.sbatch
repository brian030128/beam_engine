#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_expSTRATv9_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_expSTRATv9_llama1b_tp1_%j.out
#SBATCH --error=slurm/logs/paper_expSTRATv9_llama1b_tp1_%j.err

# expSTRAT v9 — TP=1 + full-fp8 on RedHatAI/Llama-3.2-1B-Instruct-FP8.
# Hypothesis: shrinking the base per-step time (1B has 16 layers and
# small hidden, so per-step ≈ 3 ms vs 17 ms on 8B) makes DT's
# ~constant per-step launch+merge overhead a bigger fraction of the
# total, widening the 1POOL-win regime from the ~7% ceiling seen on
# 8B. Theoretical ceiling on 1B is ~15-20%.
#
# Same shapes as v6 + v7 + one BR1 control to also see the DT-win
# side on 1B.
#
#   tag | scenario              | K  | B  | mn  | R    | regime
#   ----+-----------------------+----+----+-----+------+----------
#   LA1 | multi_few_shot        |  2 |  2 |  32 |    4 | tiny tail
#   LA2 | multi_few_shot        |  2 |  4 |  32 |    8 | tiny tail
#   LA3 | multi_few_shot        |  4 |  2 |  32 |    8 | tiny tail
#   LA4 | multi_chain_reasoning |  2 |  4 |  32 |    8 | L_priv=0
#   LA5 | multi_chain_reasoning |  2 |  2 |  32 |    4 | L_priv=0
#   LA7 | multi_chain_reasoning |  2 |  1 |  32 |    2 | minimum R
#   BR1 | multi_chain_reasoning | 64 | 32 | 256 | 2048 | DT-win control

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME=/work/u4320956/hf
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="RedHatAI/Llama-3.2-1B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3

MODEL_TAG="Llama-3.2-1B-FP8"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/expSTRATv9/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== expSTRATv9 ${MODEL_TAG} TP=1 started $(date -Iseconds) ==="
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

run_cell LA1 multi_few_shot         2  2  32
run_cell LA2 multi_few_shot         2  4  32
run_cell LA3 multi_few_shot         4  2  32
run_cell LA4 multi_chain_reasoning  2  4  32
run_cell LA5 multi_chain_reasoning  2  2  32
run_cell LA7 multi_chain_reasoning  2  1  32
run_cell BR1 multi_chain_reasoning 64 32 256

{
    echo
    echo "=== done $(date -Iseconds) ==="
} | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_smBK_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/paper_smBK_8b_%j.out
#SBATCH --error=slurm/logs/paper_smBK_8b_%j.err

# Small-(B, K) probe on Llama-3.1-8B to find cells where the picker
# should pick SHARED_2L_1POOL instead of SHARED_2L_DEC_TAIL. The current
# end-to-end runs show DEC_TAIL dominating at K=64; reduce K and/or B
# to shrink the per-branch-tail aggregate KV work and force the picker
# toward FUSED.
#
# Sweep cells (multi_chain_reasoning, --no-stage2, K and B varied):
#   (B=1, K=64), (B=1, K=16), (B=1, K=4),
#   (B=8, K=16), (B=8, K=4),
#   (B=4, K=64).
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp1c_v2_smallBK/Llama-3.1-8B/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="meta-llama/Llama-3.1-8B"

MODEL_TAG="Llama-3.1-8B"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1c_v2_smallBK/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== smallBK probe ${MODEL_TAG} started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=1)"
    echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo
} | tee "$LOG"

SGLANG_METHODS="bs_kernel bsk_2l_1p bsk_2l_dt"

run_cell () {
    # run_cell <scenario> <B> <K>
    local scenario="$1"; shift
    local B="$1"; shift
    local K="$1"; shift
    local csv="$OUT_DIR/${scenario}_B${B}_K${K}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== cell: ${scenario}  (B=${B}, K=${K}, max_new=256) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $SGLANG_METHODS \
        --paper-exact --no-stage2 \
        --max-new 256 --repeat 1 --warmup \
        --scenarios "$scenario" --k-override "$K" --b-override "$B" \
        --out "$csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (cell $scenario B=$B K=$K failed; continuing)" | tee -a "$LOG"
}

# multi_chain_reasoning sweep — small B and small K
run_cell multi_chain_reasoning 1 64
run_cell multi_chain_reasoning 1 16
run_cell multi_chain_reasoning 1  4
run_cell multi_chain_reasoning 8 16
run_cell multi_chain_reasoning 8  4
run_cell multi_chain_reasoning 4 64

# multi_few_shot sweep — should also flip to FUSED at small K
run_cell multi_few_shot        1 64
run_cell multi_few_shot        8  4

# multi_level_system sweep
run_cell multi_level_system    1 64
run_cell multi_level_system    4  4

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

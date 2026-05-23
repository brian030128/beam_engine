#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=verify_ml_8b_b3b4b5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/verify_ml_8b_b3b4b5_%j.out
#SBATCH --error=slurm/logs/verify_ml_8b_b3b4b5_%j.err

# Regression check after the multi-level TreeSpec landing
# (project_multilevel_tree_spec.md / docs/multilevel_tree_spec.md).
# Re-runs picker_demo cells B3/B4/B5 on Llama-3.1-8B-Instruct bf16
# with all 6 methods in the user's reference table:
#   paged | fasttree | deft | bs_kernel (picker) | bsk_2l_dt | bsk_2l_1p_single
# Uses legacy 2-level TreeSpec (no --tree-root) so we can verify the
# additive change doesn't regress prior numbers.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME=/work/u4320956/hf
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16

MODEL_TAG="Llama-3.1-8B-bf16"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/verify_multilevel/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== verify multilevel landing: B3/B4/B5 | ${MODEL_TAG} | $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=1, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo "purpose:  legacy-path regression check after TreeSpec.root landing"
    echo "git head: $(git rev-parse --short HEAD)"
    echo "branch:   $(git rev-parse --abbrev-ref HEAD)"
} | tee -a "$LOG"

# Use existing coefficient cache (~/.cache/beam_engine/coeffs-*.json
# from 2026-05-23 morning); skip recalibration.

SGLANG_METHODS="paged fasttree deft bs_kernel bsk_2l_dt bsk_2l_1p_single"

run_sglang_cell () {
    local tag="$1" scenario="$2" K="$3" B="$4" mn="$5"
    local csv="$OUT_DIR/${tag}_${scenario}_K${K}_B${B}_mn${mn}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== ${tag}: ${scenario}  (K=${K}, B=${B}, mn=${mn}, R=$((B*K))) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    rm -f "$csv"
    torchrun --standalone --nproc_per_node=1 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $SGLANG_METHODS \
        --paper-exact --no-stage2 \
        --max-new "$mn" --repeat 3 --warmup \
        --scenarios "$scenario" --k-override "$K" --b-override "$B" \
        --out "$csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (cell $tag failed; continuing)" | tee -a "$LOG"
}

run_sglang_cell B3 multi_doc_qa         64 1 256
run_sglang_cell B4 multi_few_shot       32 4 256
run_sglang_cell B5 multi_level_system   64 4 256

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

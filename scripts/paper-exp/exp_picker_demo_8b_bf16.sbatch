#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_picker_demo_8b_bf16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:30:00
#SBATCH --output=slurm/logs/picker_demo_8b_bf16_%j.out
#SBATCH --error=slurm/logs/picker_demo_8b_bf16_%j.err

# Picker-demo cross-precision rerun, Model B (Llama-3.1-8B-Instruct,
# bf16 weights + bf16 KV). Mirrors the B-series cells B1-B5 of the
# fp8 main run, with non-quantized weights. B6 (K=64 B=32) is excluded
# because the KV cache for R=2048 over multi_chain_reasoning won't
# fit even under bf16 weights (already OOMs at fp8 weights + bf16 KV).
# 8B in bf16 = ~16 GiB weights, leaves ~62 GiB for KV — sufficient for
# B1-B5.

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
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16

MODEL_TAG="Llama-3.1-8B-bf16"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/picker_demo/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== picker_demo Model B bf16 (${MODEL_TAG} TP=1 bf16 weights + bf16 KV) started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=1, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo
} | tee -a "$LOG"

echo "=== recalibrating coefficients (bf16) ===" | tee -a "$LOG"
uv run python -m beam_engine.methods.bs_kernel.calibrate --force 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"

SGLANG_ALL_METHODS="paged fasttree deft bs_kernel bsk_2l_1p bsk_2l_dt"
BATCHED_ALL_METHODS="paged fasttree deft bs_kernel bs_kernel_2l1p bs_kernel_2ldt"

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
        --methods $SGLANG_ALL_METHODS \
        --paper-exact --no-stage2 \
        --max-new "$mn" --repeat 3 --warmup \
        --scenarios "$scenario" --k-override "$K" --b-override "$B" \
        --out "$csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (cell $tag failed; continuing)" | tee -a "$LOG"
}

run_batched_cell () {
    local tag="$1" K="$2" B="$3" Lp="$4" mn="$5"
    local csv="$OUT_DIR/${tag}_beam_search_K${K}_B${B}_Lp${Lp}_mn${mn}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== ${tag}: beam_search (hotpot)  (K=${K}, B=${B}, L_p=${Lp}, mn=${mn}) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    local first=1
    for m in $BATCHED_ALL_METHODS; do
        echo "  --- method: $m ---" | tee -a "$LOG"
        local tmp="$OUT_DIR/${tag}_${m}.csv"
        uv run python benchmarks/bs_kernel/bench_batched.py \
            --K "$K" --L_p "$Lp" --B "$B" --max_new "$mn" \
            --prompts-file data/hotpotqa.jsonl \
            --warmup --methods "$m" --out "$tmp" \
            2>&1 | tee -a "$LOG" \
            || { echo "  ($m failed; continuing)" | tee -a "$LOG"; continue; }
        if [[ $first -eq 1 ]]; then
            [[ -f "$tmp" ]] && cp "$tmp" "$csv" && first=0
        else
            [[ -f "$tmp" ]] && tail -n +2 "$tmp" >> "$csv"
        fi
    done
}

# Low-batch B-series cells (B1-B5). B6 (B=32) skipped — won't fit.
run_batched_cell B1                            4 1 40000 128
run_sglang_cell  B2 multi_chain_reasoning      4 1 32
run_sglang_cell  B3 multi_doc_qa              64 1 256
run_sglang_cell  B4 multi_few_shot            32 4 256
run_sglang_cell  B5 multi_level_system        64 4 256

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_picker_demo_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/picker_demo_70b_%j.out
#SBATCH --error=slurm/logs/picker_demo_70b_%j.err

# Picker-demo experiment, Model C (Llama-3-70B-Instruct-FP8 TP=2
# fp8+fp8). Cells C1-C6 and D1-D4 from docs/exp_picker_demo.md.
#
# Smart-rerun mode (cost-model refresh):
#   - If the cell CSV already has baseline rows, only refresh bs_kernel
#     methods. Otherwise run all six methods.
#   - On C3-C6 the existing CSVs only carry paged + bs_kernel — fasttree
#     / deft don't yet support fp8 KV append on 70B, so the bsk_only
#     path is the right one anyway.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# 70B fp8 checkpoint is ~70 GiB on disk; use ephemeral SLURM tmp,
# trap-rm on EXIT (per feedback_large_model_ephemeral_cache.md).
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3

MODEL_TAG="Llama-3-70B-Instruct-FP8"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/picker_demo/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== picker_demo Model C (${MODEL_TAG} TP=2 fp8+fp8) started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo "mode:     smart-rerun (refresh bs_kernel against new cost model)"
    echo
} | tee -a "$LOG"

echo "=== recalibrating coefficients ===" | tee -a "$LOG"
uv run python -m beam_engine.methods.bs_kernel.calibrate --force 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"

SGLANG_BSK_METHODS="bs_kernel bsk_2l_1p bsk_2l_dt"
SGLANG_ALL_METHODS="paged fasttree deft bs_kernel bsk_2l_1p bsk_2l_dt"
BATCHED_BSK_METHODS="bs_kernel bs_kernel_2l1p bs_kernel_2ldt"
BATCHED_ALL_METHODS="paged fasttree deft bs_kernel bs_kernel_2l1p bs_kernel_2ldt"

run_sglang_cell () {
    local tag="$1"; shift
    local scenario="$1"; shift
    local K="$1"; shift
    local B="$1"; shift
    local mn="$1"; shift
    local csv="$OUT_DIR/${tag}_${scenario}_K${K}_B${B}_mn${mn}.csv"
    local tmp="$OUT_DIR/${tag}_${scenario}_K${K}_B${B}_mn${mn}.bsk.csv"

    local decision_kv
    decision_kv=$(uv run python scripts/paper-exp/decide_methods.py "$csv" 2>/dev/null || echo "all bf16")
    local decision="${decision_kv% *}"
    local methods out
    if [[ "$decision" == "bsk_only" ]]; then
        methods="$SGLANG_BSK_METHODS"; out="$tmp"
    else
        methods="$SGLANG_ALL_METHODS"; out="$csv"
    fi
    {
        echo
        echo "=========================================================="
        echo "=== ${tag}: ${scenario}  (K=${K}, B=${B}, mn=${mn}, R=$((B*K)))  decision=${decision_kv} ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $methods \
        --paper-exact --no-stage2 \
        --max-new "$mn" --repeat 3 --warmup \
        --scenarios "$scenario" --k-override "$K" --b-override "$B" \
        --out "$out" \
        2>&1 | tee -a "$LOG" \
        || echo "  (cell $tag failed; continuing)" | tee -a "$LOG"
    if [[ "$decision" == "bsk_only" && -f "$tmp" ]]; then
        uv run python scripts/paper-exp/merge_picker_csv.py "$csv" "$tmp" 2>&1 | tee -a "$LOG"
        rm -f "$tmp"
    fi
}

run_batched_cell () {
    local tag="$1"; shift
    local K="$1"; shift
    local B="$1"; shift
    local Lp="$1"; shift
    local mn="$1"; shift
    local csv="$OUT_DIR/${tag}_beam_search_K${K}_B${B}_Lp${Lp}_mn${mn}.csv"

    local decision_kv
    decision_kv=$(uv run python scripts/paper-exp/decide_methods.py "$csv" 2>/dev/null || echo "all bf16")
    local decision="${decision_kv% *}"
    local methods_list
    if [[ "$decision" == "bsk_only" ]]; then
        methods_list="$BATCHED_BSK_METHODS"
    else
        methods_list="$BATCHED_ALL_METHODS"
    fi
    {
        echo
        echo "=========================================================="
        echo "=== ${tag}: beam_search (hotpot)  (K=${K}, B=${B}, L_p=${Lp}, mn=${mn})  decision=${decision_kv} ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    local first=1
    local tmp_combined="$OUT_DIR/${tag}_beam_search_K${K}_B${B}_Lp${Lp}_mn${mn}.bsk.csv"
    local out_target
    if [[ "$decision" == "bsk_only" ]]; then out_target="$tmp_combined"; else out_target="$csv"; fi
    for m in $methods_list; do
        echo "  --- method: $m ---" | tee -a "$LOG"
        local tmp="$OUT_DIR/${tag}_${m}.csv"
        torchrun --standalone --nproc_per_node=2 \
            benchmarks/bs_kernel/bench_batched.py \
            --K "$K" --L_p "$Lp" --B "$B" --max_new "$mn" \
            --prompts-file data/hotpotqa.jsonl \
            --warmup --methods "$m" --out "$tmp" \
            2>&1 | tee -a "$LOG" \
            || { echo "  ($m failed; continuing)" | tee -a "$LOG"; continue; }
        if [[ $first -eq 1 ]]; then
            [[ -f "$tmp" ]] && cp "$tmp" "$out_target" && first=0
        else
            [[ -f "$tmp" ]] && tail -n +2 "$tmp" >> "$out_target"
        fi
    done
    if [[ "$decision" == "bsk_only" && -f "$tmp_combined" ]]; then
        uv run python scripts/paper-exp/merge_picker_csv.py "$csv" "$tmp_combined" 2>&1 | tee -a "$LOG"
        rm -f "$tmp_combined"
    fi
}

run_batched_cell C1                            4  1 40000 64
run_sglang_cell  C2 multi_chain_reasoning      4  1  32
run_sglang_cell  C3 multi_doc_qa              32  1 256
run_sglang_cell  C4 multi_few_shot            16  4 256
run_sglang_cell  C5 multi_level_system        32  4 256
run_sglang_cell  C6 multi_chain_reasoning     64  4 256

# Large-batch D-series.
run_sglang_cell  D1 multi_chain_reasoning     32 32 256
run_sglang_cell  D2 multi_few_shot            16 32 256
run_sglang_cell  D3 multi_level_system        32 16 256
run_sglang_cell  D4 multi_doc_qa              16  4 256

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

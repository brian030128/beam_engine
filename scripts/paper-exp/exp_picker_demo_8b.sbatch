#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_picker_demo_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/picker_demo_8b_%j.out
#SBATCH --error=slurm/logs/picker_demo_8b_%j.err

# Picker-demo experiment, Model B (Llama-3.1-8B-FP8 TP=1 fp8 weights,
# bf16 KV by default). Cells B1-B6 and D1-D4 from
# docs/exp_picker_demo.md.
#
# Smart-rerun mode (cost-model refresh):
#   - If the cell CSV already has baseline rows (paged/fasttree/deft),
#     only re-runs the bs_kernel methods and merges them in.
#   - Otherwise runs all six methods.
#   - On OOM under bf16 KV, retries the same cell under fp8 KV with
#     paged + bs_kernel only (fasttree/deft can't append fp8 KV yet).
#     If both fail, gives up on that cell.

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
export BE_MODEL="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"
export BE_DTYPE=bf16

MODEL_TAG="Llama-3.1-8B-FP8"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/picker_demo/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== picker_demo Model B (${MODEL_TAG} TP=1 fp8 weights, bf16 KV) started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=1, dtype=$BE_DTYPE, kv=bf16; fp8-KV fallback on OOM)"
    echo "mode:     smart-rerun (refresh bs_kernel against new cost model)"
    echo
} | tee -a "$LOG"

echo "=== recalibrating coefficients ===" | tee -a "$LOG"
uv run python -m beam_engine.methods.bs_kernel.calibrate --force 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"

SGLANG_BSK_METHODS="bs_kernel bsk_2l_1p bsk_2l_dt"
SGLANG_ALL_METHODS="paged fasttree deft bs_kernel bsk_2l_1p bsk_2l_dt"
SGLANG_FP8KV_METHODS_ALL="paged bs_kernel bsk_2l_1p bsk_2l_dt"
BATCHED_BSK_METHODS="bs_kernel bs_kernel_2l1p bs_kernel_2ldt"
BATCHED_ALL_METHODS="paged fasttree deft bs_kernel bs_kernel_2l1p bs_kernel_2ldt"

_n_rows () {
    [[ -f "$1" ]] && echo $(($(wc -l < "$1") - 1)) || echo 0
}

_run_sglang () {
    local methods="$1" scenario="$2" K="$3" B="$4" mn="$5" out="$6" kv_dtype="${7:-}"
    if [[ -n "$kv_dtype" ]]; then
        BE_KV_DTYPE="$kv_dtype" torchrun --standalone --nproc_per_node=1 \
            benchmarks/bs_kernel/bench_sglang_e2e.py \
            --methods $methods --paper-exact --no-stage2 \
            --max-new "$mn" --repeat 3 --warmup \
            --scenarios "$scenario" --k-override "$K" --b-override "$B" \
            --out "$out" 2>&1 | tee -a "$LOG"
    else
        torchrun --standalone --nproc_per_node=1 \
            benchmarks/bs_kernel/bench_sglang_e2e.py \
            --methods $methods --paper-exact --no-stage2 \
            --max-new "$mn" --repeat 3 --warmup \
            --scenarios "$scenario" --k-override "$K" --b-override "$B" \
            --out "$out" 2>&1 | tee -a "$LOG"
    fi
}

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
    local dtype="${decision_kv##* }"
    local methods out kv=""
    if [[ "$decision" == "bsk_only" ]]; then
        methods="$SGLANG_BSK_METHODS"; out="$tmp"
        [[ "$dtype" == "fp8" ]] && kv="fp8_e4m3"
    else
        methods="$SGLANG_ALL_METHODS"; out="$csv"
    fi
    {
        echo
        echo "=========================================================="
        echo "=== ${tag}: ${scenario}  (K=${K}, B=${B}, mn=${mn}, R=$((B*K)))  decision=${decision} dtype=${dtype} ==="
        echo "=========================================================="
    } | tee -a "$LOG"

    rm -f "$out"
    _run_sglang "$methods" "$scenario" "$K" "$B" "$mn" "$out" "$kv"
    # bench_sglang_e2e.py swallows per-method OOMs and exits 0; detect
    # zero-row output and retry under fp8 KV.
    if [[ $(_n_rows "$out") -eq 0 ]]; then
        echo "  ($tag produced no rows under '${dtype:-bf16}' KV; retrying with fp8 KV)" | tee -a "$LOG"
        local fp8_methods="$SGLANG_FP8KV_METHODS_ALL"
        [[ "$decision" == "bsk_only" ]] && fp8_methods="$SGLANG_BSK_METHODS"
        _run_sglang "$fp8_methods" "$scenario" "$K" "$B" "$mn" "$out" "fp8_e4m3"
        if [[ $(_n_rows "$out") -eq 0 ]]; then
            echo "  ($tag failed under both bf16 and fp8 KV; giving up)" | tee -a "$LOG"
            return
        fi
    fi
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

    local decision
    decision=$(uv run python scripts/paper-exp/decide_methods.py "$csv" 2>/dev/null || echo all)
    local methods_list
    if [[ "$decision" == "bsk_only" ]]; then
        methods_list="$BATCHED_BSK_METHODS"
    else
        methods_list="$BATCHED_ALL_METHODS"
    fi
    {
        echo
        echo "=========================================================="
        echo "=== ${tag}: beam_search (hotpot)  (K=${K}, B=${B}, L_p=${Lp}, mn=${mn})  decision=${decision} ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    local first=1
    local tmp_combined="$OUT_DIR/${tag}_beam_search_K${K}_B${B}_Lp${Lp}_mn${mn}.bsk.csv"
    local out_target
    if [[ "$decision" == "bsk_only" ]]; then out_target="$tmp_combined"; else out_target="$csv"; fi
    for m in $methods_list; do
        echo "  --- method: $m ---" | tee -a "$LOG"
        local tmp="$OUT_DIR/${tag}_${m}.csv"
        uv run python benchmarks/bs_kernel/bench_batched.py \
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

run_batched_cell B1                            4  1 40000 128
run_sglang_cell  B2 multi_chain_reasoning      4  1  32
run_sglang_cell  B3 multi_doc_qa              64  1 256
run_sglang_cell  B4 multi_few_shot            32  4 256
run_sglang_cell  B5 multi_level_system        64  4 256
run_sglang_cell  B6 multi_chain_reasoning     64 32 256

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

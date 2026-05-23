#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_pd_70b_2l1ps_vs_2ldt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/picker_demo_70b_2l1ps_vs_2ldt_%j.out
#SBATCH --error=slurm/logs/picker_demo_70b_2l1ps_vs_2ldt_%j.err

# Same cells as exp_picker_demo_70b.sbatch (Model C+D, C1-C6, D1-D4) but
# restricted to bs_kernel_2l1p_single + bs_kernel_2ldt (batched) and
# bsk_2l_1p_single + bsk_2l_dt (sglang). Results -> picker_demo_2l1p_single_vs_2ldt/.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# Ephemeral HF cache on SLURM tmp (70B fp8 = ~70 GiB).
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
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/picker_demo_2l1p_single_vs_2ldt/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== picker_demo 2l1p_single vs 2ldt | Model C+D (${MODEL_TAG}) | $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo "flashinfer diff (single_launch must be live):"
    git -C 3rdparty/flashinfer diff --stat 2>&1 | sed 's/^/  /'
} | tee -a "$LOG"

SGLANG_METHODS="bsk_2l_1p_single bsk_2l_dt"
BATCHED_METHODS="bs_kernel_2l1p_single bs_kernel_2ldt"

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
    local first=1
    for m in $SGLANG_METHODS; do
        local tmp="$OUT_DIR/${tag}_${m}.csv"
        echo "  --- method: $m ---" | tee -a "$LOG"
        torchrun --standalone --nproc_per_node=2 \
            benchmarks/bs_kernel/bench_sglang_e2e.py \
            --methods "$m" \
            --paper-exact --no-stage2 \
            --max-new "$mn" --repeat 3 --warmup \
            --scenarios "$scenario" --k-override "$K" --b-override "$B" \
            --out "$tmp" \
            2>&1 | tee -a "$LOG" \
            || { echo "  ($m failed; continuing)" | tee -a "$LOG"; continue; }
        if [[ $first -eq 1 ]]; then
            [[ -f "$tmp" ]] && cp "$tmp" "$csv" && first=0
        else
            [[ -f "$tmp" ]] && tail -n +2 "$tmp" >> "$csv"
        fi
    done
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
    for m in $BATCHED_METHODS; do
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
            [[ -f "$tmp" ]] && cp "$tmp" "$csv" && first=0
        else
            [[ -f "$tmp" ]] && tail -n +2 "$tmp" >> "$csv"
        fi
    done
}

# C series (low/medium batch) and D series (large batch).
run_batched_cell C1                            4  1 40000 64
run_sglang_cell  C2 multi_chain_reasoning      4  1  32
run_sglang_cell  C3 multi_doc_qa              32  1 256
run_sglang_cell  C4 multi_few_shot            16  4 256
run_sglang_cell  C5 multi_level_system        32  4 256
run_sglang_cell  C6 multi_chain_reasoning     64  4 256
run_sglang_cell  D1 multi_chain_reasoning     32 32 256
run_sglang_cell  D2 multi_few_shot            16 32 256
run_sglang_cell  D3 multi_level_system        32 16 256
run_sglang_cell  D4 multi_doc_qa              16  4 256

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

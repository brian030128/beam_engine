#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_70b_b3b4b5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/pe_70b_b3b4b5_%j.out
#SBATCH --error=slurm/logs/pe_70b_b3b4b5_%j.err

# Run 8B's B3/B4/B5 cells on Llama-3-70B-Instruct-FP8 (TP=2, fp8 weights
# + fp8 KV), full method set (paged, fasttree, deft, bs_kernel picker,
# bsk_2l_1p_single, bsk_2l_dt). Results land in a new dir so existing
# 70B picker_demo CSVs stay untouched.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# 70B fp8 checkpoint is ~70 GiB on disk; ephemeral HF cache on SLURM tmp.
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
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/picker_demo_2l1p_single_vs_2ldt/${MODEL_TAG}_8b_b3b4b5"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== 70B fp8 @ 8B B3/B4/B5 cells | $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo "flashinfer diff (single_launch must be live):"
    git -C 3rdparty/flashinfer diff --stat 2>&1 | sed 's/^/  /'
} | tee -a "$LOG"

# Calibrate so the bs_kernel picker has the right coefficients for fp8
# weights + fp8 KV on TP=2 (picker_demo cells call this; copying for
# consistency).
echo "=== recalibrating coefficients (bf16 act + fp8 KV) ===" | tee -a "$LOG"
uv run python -m beam_engine.methods.bs_kernel.calibrate --force 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"

# Each forced method run in its own torchrun process so the
# single_launch monkey-patch in bsk_2l_1p_single doesn't leak.
SGLANG_METHODS="paged fasttree deft bs_kernel bsk_2l_1p_single bsk_2l_dt"

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

# B3/B4/B5 cells from the 8B paper run.
run_sglang_cell  B3 multi_doc_qa             64  1 256
run_sglang_cell  B4 multi_few_shot           32  4 256
run_sglang_cell  B5 multi_level_system       64  4 256

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

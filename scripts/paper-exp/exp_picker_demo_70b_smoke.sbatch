#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_picker_demo_70b_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/picker_demo_70b_smoke_%j.out
#SBATCH --error=slurm/logs/picker_demo_70b_smoke_%j.err

# Smoke test for the small-BS decode-probe extension in calibrate.py.
# Goal: see whether adding (32,128), (32,256), (64,128), (64,256) to
# the decode regression flips the C3 picker (was 1.190× of oracle,
# picked 1POOL when DT was 19% faster) without breaking C6 (currently
# correct DT pick).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

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
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/picker_demo/${MODEL_TAG}/smoke_v3"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== smoke v3: 70B C3+C6 w/ extended decode probes $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo "change:   calibrate.py BS_pairs now includes (32,128) (32,256)"
    echo "          (64,128) (64,256) plus original four; OLS sees the"
    echo "          L2-resident / launch-dominated regime that C3 lives in"
    echo
} | tee "$LOG"

echo "=== recalibrating with extended decode probes ===" | tee -a "$LOG"
uv run python -m beam_engine.methods.bs_kernel.calibrate --force 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"

SGLANG_METHODS="paged bs_kernel bsk_2l_1p bsk_2l_dt"

run_sglang_cell () {
    local tag="$1"; shift; local scenario="$1"; shift
    local K="$1"; shift; local B="$1"; shift; local mn="$1"; shift
    local csv="$OUT_DIR/${tag}_${scenario}_K${K}_B${B}_mn${mn}.csv"
    echo "=== ${tag}: ${scenario}  (K=${K}, B=${B}, mn=${mn}) ===" | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $SGLANG_METHODS --paper-exact --no-stage2 \
        --max-new "$mn" --repeat 3 --warmup \
        --scenarios "$scenario" --k-override "$K" --b-override "$B" \
        --out "$csv" 2>&1 | tee -a "$LOG" \
        || echo "  (cell $tag failed; continuing)" | tee -a "$LOG"
}

run_sglang_cell C3 multi_doc_qa            32 1 256   # the mispicked cell
run_sglang_cell C6 multi_chain_reasoning   64 4 256   # regression check (was DT pick ✓)

echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"

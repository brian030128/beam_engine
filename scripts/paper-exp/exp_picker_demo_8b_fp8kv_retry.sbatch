#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_picker_demo_8b_fp8kv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/picker_demo_8b_fp8kv_%j.out
#SBATCH --error=slurm/logs/picker_demo_8b_fp8kv_%j.err

# Retry-only sbatch: rerun the three picker_demo 8B cells that OOMed
# under bf16 KV (B6, D2, D4) using fp8 KV. fasttree/deft can't append
# fp8 KV yet, so the methods list is paged + bs_kernel only.

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
export BE_KV_DTYPE=fp8_e4m3

MODEL_TAG="Llama-3.1-8B-FP8"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/picker_demo/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo
    echo "=== picker_demo 8B fp8-KV retry started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=1, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo
} | tee -a "$LOG"

# bs_kernel coefficients calibrated under bf16 KV in the main job;
# don't recalibrate here (would overwrite with fp8-KV coefficients
# and shift the picker on the bf16-KV cells already completed).

METHODS="paged bs_kernel bsk_2l_1p bsk_2l_dt"

run_cell () {
    local tag="$1" scenario="$2" K="$3" B="$4" mn="$5"
    local csv="$OUT_DIR/${tag}_${scenario}_K${K}_B${B}_mn${mn}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== ${tag} (fp8-KV retry): ${scenario}  (K=${K}, B=${B}, mn=${mn}, R=$((B*K))) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=1 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS --paper-exact --no-stage2 \
        --max-new "$mn" --repeat 3 --warmup \
        --scenarios "$scenario" --k-override "$K" --b-override "$B" \
        --out "$csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (cell $tag failed; continuing)" | tee -a "$LOG"
}

run_cell B6 multi_chain_reasoning 64 32 256
run_cell D2 multi_few_shot        16 32 256
run_cell D4 multi_doc_qa          16  4 256

{
    echo
    echo "=== 8B fp8-KV retry done $(date -Iseconds) ==="
} | tee -a "$LOG"

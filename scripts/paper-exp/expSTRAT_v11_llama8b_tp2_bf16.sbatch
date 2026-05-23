#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_expSTRATv11_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_expSTRATv11_llama8b_tp2_bf16_%j.out
#SBATCH --error=slurm/logs/paper_expSTRATv11_llama8b_tp2_bf16_%j.err

# expSTRAT v11 — unquantized (bf16 weights + bf16 KV) Llama-3.1-8B on
# TP=2, single gov_report cell (B=1, K=16, L_p=40000, mn=256).
# Counterpart to v8 (TP=1 fp8 weights + fp8 KV at the same shape) so
# we can isolate the effect of dropping all quantization.
#
# Memory budget (per rank):
#   weights bf16 8B on TP=2: ~8 GiB
#   KV bf16, 4 KV heads/rank: 644k tokens × 65 KB ≈ 41 GiB
#   total ≈ 49 GiB — fits on H100 80 GiB with headroom

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME=/work/u4320956/hf
# meta-llama/Llama-3.1-8B is a gated repo; load token from the
# huggingface-cli cache if HF_TOKEN isn't already set in env.
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="meta-llama/Llama-3.1-8B"
# No BE_DTYPE / BE_KV_DTYPE export → bf16 weights and bf16 KV defaults.

MODEL_TAG="Llama-3.1-8B-bf16"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/expSTRATv11/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== expSTRATv11 ${MODEL_TAG} TP=2 bf16 started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, bf16 weights + bf16 KV)"
    echo
} | tee "$LOG"

K=16; B=1; L_p=40000; max_new=256
csv="$OUT_DIR/beam_search_K${K}_B${B}_Lp${L_p}_mn${max_new}.csv"
{
    echo "=========================================================="
    echo "=== LB1-bf16: gov_report  (K=${K}, B=${B}, L_p=${L_p}, max_new=${max_new}, tp=2, bf16) ==="
    echo "=========================================================="
} | tee -a "$LOG"

first=1
for m in paged bs_kernel bs_kernel_2l1p bs_kernel_2ldt; do
    echo "  --- method: $m ---" | tee -a "$LOG"
    tmp="$OUT_DIR/beam_search_${m}.csv"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_batched.py \
        --K "$K" --L_p "$L_p" --B "$B" --max_new "$max_new" \
        --prompts-file data/gov_report.jsonl \
        --warmup \
        --methods "$m" \
        --out "$tmp" \
        2>&1 | tee -a "$LOG" \
        || { echo "  (method $m failed; continuing)" | tee -a "$LOG"; continue; }
    if [[ $first -eq 1 ]]; then
        [[ -f "$tmp" ]] && cp "$tmp" "$csv" && first=0
    else
        [[ -f "$tmp" ]] && tail -n +2 "$tmp" >> "$csv"
    fi
done

{
    echo
    echo "=== done $(date -Iseconds) ==="
} | tee -a "$LOG"

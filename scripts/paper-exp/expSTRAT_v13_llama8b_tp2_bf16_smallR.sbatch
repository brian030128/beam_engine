#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_expSTRATv13_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_expSTRATv13_llama8b_tp2_bf16_%j.out
#SBATCH --error=slurm/logs/paper_expSTRATv13_llama8b_tp2_bf16_%j.err

# expSTRAT v13 (Probe B) — TP=2 bf16, smaller R at moderate L_p.
# Hypothesis: shrinking R amplifies DT's idle-SM tax without bloating
# the base. v11 had R=16 (1P +9%); v12 LB1a R=16 L_p=20k (1P +12%).
# v13 drops R to 4 to test if even smaller R + moderate L_p widens.
#
#   tag  | K | B | mn  | L_p   | R
#   PB1  | 4 | 1 | 128 | 20000 | 4
#   PB2  | 4 | 1 | 128 | 30000 | 4
#   PB3  | 8 | 1 | 128 | 20000 | 8

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME=/work/u4320956/hf
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="meta-llama/Llama-3.1-8B"

MODEL_TAG="Llama-3.1-8B-bf16"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/expSTRATv13/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== expSTRATv13 ${MODEL_TAG} TP=2 bf16 small-R probe started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, bf16 weights + bf16 KV)"
    echo
} | tee "$LOG"

run_cell () {
    local tag="$1"; shift
    local K="$1"; shift
    local B="$1"; shift
    local L_p="$1"; shift
    local mn="$1"; shift
    local csv="$OUT_DIR/beam_search_K${K}_B${B}_Lp${L_p}_mn${mn}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== ${tag}: gov_report  (K=${K}, B=${B}, L_p=${L_p}, max_new=${mn}, R=$((K*B)), tp=2, bf16) ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    local first=1
    for m in paged bs_kernel bs_kernel_2l1p bs_kernel_2ldt; do
        echo "  --- method: $m ---" | tee -a "$LOG"
        local tmp="$OUT_DIR/beam_search_${tag}_${m}.csv"
        torchrun --standalone --nproc_per_node=2 \
            benchmarks/bs_kernel/bench_batched.py \
            --K "$K" --L_p "$L_p" --B "$B" --max_new "$mn" \
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
}

run_cell PB1 4 1 20000 128
run_cell PB2 4 1 30000 128
run_cell PB3 8 1 20000 128

{
    echo
    echo "=== done $(date -Iseconds) ==="
} | tee -a "$LOG"

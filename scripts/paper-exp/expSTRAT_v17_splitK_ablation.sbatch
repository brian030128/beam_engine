#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_expSTRATv17_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_expSTRATv17_%j.out
#SBATCH --error=slurm/logs/paper_expSTRATv17_%j.err

# expSTRAT v17 — split-K ablation on the v11 cell (K=16 B=1 L_p=40k
# mn=256 TP=2 bf16). Test whether the cascade is over-splitting L0:
# - baseline: heuristic split (currently splits L0 into 16 chunks)
# - no_L0:    force L0 num_chunks=1 via FLASHINFER_FORCE_NO_SPLIT_LEVELS=0
#
# At v11, L1 already produces 128 CTAs (16 requests × 8 kv_heads). If
# L0 + L1 co-schedule on Pool_T16, L1 alone may already saturate the
# 132 SMs, making L0's split-K pure merge overhead.

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
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/expSTRATv17/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== expSTRATv17 split-K ablation $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, bf16)"
    echo "shape:    K=16 B=1 L_p=40000 mn=256 (gov_report)"
    echo
} | tee "$LOG"

K=16; B=1; L_p=40000; max_new=256

run_variant () {
    local variant="$1"; shift
    local force_no_split="$1"; shift
    local csv="$OUT_DIR/beam_search_${variant}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== ${variant}  (FLASHINFER_FORCE_NO_SPLIT_LEVELS='${force_no_split}') ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    local first=1
    for m in paged bs_kernel bs_kernel_2l1p bs_kernel_2ldt; do
        echo "  --- method: $m ---" | tee -a "$LOG"
        local tmp="$OUT_DIR/beam_search_${variant}_${m}.csv"
        FLASHINFER_FORCE_NO_SPLIT_LEVELS="$force_no_split" \
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
}

run_variant baseline ""
run_variant no_L0   "0"

{
    echo
    echo "=== done $(date -Iseconds) ==="
} | tee -a "$LOG"

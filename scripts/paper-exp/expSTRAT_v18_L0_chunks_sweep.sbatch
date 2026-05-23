#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_expSTRATv18_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/paper_expSTRATv18_%j.out
#SBATCH --error=slurm/logs/paper_expSTRATv18_%j.err

# expSTRAT v18 — sweep L0 split-K chunk count on the v11 cell to find
# the sweet spot between (a) parallelism + L2 reuse via more chunks
# and (b) merge_states launch overhead from over-splitting.
#
# Baseline heuristic at v11 (K=16 B=1 L_p=40k mn=256 TP=2 bf16):
#   - L0 splits into 16 chunks (chunk_size ≈ 2496 tokens)
#   - L1 doesn't split (qo_tiles_l=16 already saturates SMs)
#
# Sweep L0 chunks ∈ {1, 2, 4, 8, 16-heuristic}.

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
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/expSTRATv18/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== expSTRATv18 L0 chunk-count sweep $(date -Iseconds) ==="
    echo "shape:    K=16 B=1 L_p=40000 mn=256 gov_report (TP=2 bf16)"
    echo
} | tee "$LOG"

K=16; B=1; L_p=40000; max_new=256

run_variant () {
    local label="$1"; shift
    local force_chunks="$1"; shift  # "" → heuristic; "0:N" → force L0 to N chunks
    local csv="$OUT_DIR/beam_search_${label}.csv"
    {
        echo
        echo "=========================================================="
        echo "=== ${label}  (FLASHINFER_FORCE_KV_CHUNKS='${force_chunks}') ==="
        echo "=========================================================="
    } | tee -a "$LOG"
    local first=1
    for m in paged bs_kernel bs_kernel_2l1p bs_kernel_2ldt; do
        echo "  --- method: $m ---" | tee -a "$LOG"
        local tmp="$OUT_DIR/beam_search_${label}_${m}.csv"
        FLASHINFER_FORCE_KV_CHUNKS="$force_chunks" \
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

# Heuristic baseline (matches v11): L0 chunks = 16
run_variant heuristic ""
# Sweep forced overrides
run_variant L0_1  "0:1"
run_variant L0_2  "0:2"
run_variant L0_4  "0:4"
run_variant L0_8  "0:8"
run_variant L0_32 "0:32"

{
    echo
    echo "=== done $(date -Iseconds) ==="
} | tee -a "$LOG"

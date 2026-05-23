#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_expSTRATv8_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_expSTRATv8_llama8b_tp1_%j.out
#SBATCH --error=slurm/logs/paper_expSTRATv8_llama8b_tp1_%j.err

# expSTRAT v8 — single long-prefix beam-search cell on
# Llama-3.1-8B-FP8 TP=1 via bench_batched.py + gov_report prompts.
# Question: does very long shared prefix (L_p≈40k) at B=1 K=16 produce
# a 1POOL ≥30% regime that mcr's L_p≈4096 cells can't reach?
# (gov_report tops out at 43019 tokens — 4 prompts ≥40k, enough for
# B=1. Original target was L_p=50k but no prompts that long.)
#
# Shape: K=16, B=1, L_p=40000, max_new=256.
# Memory: 1×16×(40000+256) ≈ 640k tokens → ~40 GiB KV pool at
# 64 KB/tok (fp8, no TP) plus ~8 GiB weights → comfortably under
# 80 GiB.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME=/work/u4320956/hf
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3

MODEL_TAG="Llama-3.1-8B-FP8"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/expSTRATv8/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== expSTRATv8 ${MODEL_TAG} TP=1 started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=1, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo
} | tee "$LOG"

K=16; B=1; L_p=40000; max_new=256
csv="$OUT_DIR/beam_search_K${K}_B${B}_Lp${L_p}_mn${max_new}.csv"
{
    echo "=========================================================="
    echo "=== LB1: gov_report  (K=${K}, B=${B}, L_p=${L_p}, max_new=${max_new}, tp=1, fp8) ==="
    echo "=========================================================="
} | tee -a "$LOG"

first=1
for m in paged bs_kernel bs_kernel_2l1p bs_kernel_2ldt; do
    echo "  --- method: $m ---" | tee -a "$LOG"
    tmp="$OUT_DIR/beam_search_${m}.csv"
    torchrun --standalone --nproc_per_node=1 \
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

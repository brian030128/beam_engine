#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_expSTRATv16_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_expSTRATv16_llama8b_tp2_bf16_%j.out
#SBATCH --error=slurm/logs/paper_expSTRATv16_llama8b_tp2_bf16_%j.err

# expSTRAT v16 — hotpotqa.jsonl at the v11 cell shape on Llama-3.1-8B
# bf16 TP=2. Tests whether v11's "1P-win ~9%, bs/paged ~0.93" survives
# moving from gov_report's monolithic long doc to a multi-document RAG
# prefix (~50 distinct HotpotQA paragraphs concatenated + a question).
#
# Shape: K=16, B=1, L_p=40000, mn=256 (matches v11 LB1).

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
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/expSTRATv16/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== expSTRATv16 ${MODEL_TAG} TP=2 hotpotqa started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL  (tp=2, bf16 weights + bf16 KV)"
    echo "prompts:  data/hotpotqa.jsonl"
    echo
} | tee "$LOG"

K=16; B=1; L_p=40000; max_new=256
csv="$OUT_DIR/beam_search_K${K}_B${B}_Lp${L_p}_mn${max_new}.csv"
{
    echo "=========================================================="
    echo "=== HQ1: hotpotqa  (K=${K}, B=${B}, L_p=${L_p}, max_new=${max_new}, tp=2, bf16) ==="
    echo "=========================================================="
} | tee -a "$LOG"

first=1
for m in paged bs_kernel bs_kernel_2l1p bs_kernel_2ldt; do
    echo "  --- method: $m ---" | tee -a "$LOG"
    tmp="$OUT_DIR/beam_search_${m}.csv"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_batched.py \
        --K "$K" --L_p "$L_p" --B "$B" --max_new "$max_new" \
        --prompts-file data/hotpotqa.jsonl \
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

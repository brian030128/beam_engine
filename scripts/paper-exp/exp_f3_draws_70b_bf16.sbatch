#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f3_draws_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/f3_draws_70b_%j.out
#SBATCH --error=slurm/logs/f3_draws_70b_%j.err

# F3 beam_search error bar — Llama-3-70B-Instruct (bf16/bf16, TP=4, B=8).
# 3 data draws (seeds 0,1,2): distinct 8-prompt windows from the HotpotQA
# pool; L_p=16000 pins the shape. --warmup discards the first iter per draw.
# All 7 methods share one process per draw (model loaded once per draw — the
# whole reason for not splitting methods into separate processes here).
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

# bf16 70B is ~140 GiB → ephemeral cache, trap-removed on exit.
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"; trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_WORKSPACE_MB=1024
export BE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16

MODEL_TAG="Llama-3-70B-bf16"; NPROC=4; B=8
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/final_paper/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run_f3_draws.log"
CSV="$OUT_DIR/F3_beam_search_K64_B${B}_Lp16000_mn256_draws012.csv"
echo "=== F3 3-draw error bar (${MODEL_TAG}) $(date -Iseconds) ===" | tee -a "$LOG"

rm -f "$CSV"; first=1
for s in 0 1 2; do
    echo "--- data_seed=$s ---" | tee -a "$LOG"
    tmp="$OUT_DIR/_f3_draws_s${s}.csv"; rm -f "$tmp"
    torchrun --standalone --nproc_per_node="$NPROC" \
        benchmarks/bs_kernel/bench_batched.py \
        --K 64 --L_p 16000 --B "$B" --max_new 256 \
        --prompts-file data/hotpotqa_f3_pool.jsonl --data-seed "$s" \
        --warmup \
        --methods paged fasttree deft mlca bs_kernel bs_kernel_2l1p bs_kernel_2ldt \
        --out "$tmp" 2>&1 | tee -a "$LOG" || echo "  (seed $s failed)" | tee -a "$LOG"
    if [[ -f "$tmp" ]]; then
        if [[ $first -eq 1 ]]; then cp "$tmp" "$CSV"; first=0; else tail -n +2 "$tmp" >> "$CSV"; fi
        rm -f "$tmp"
    fi
done
echo "=== done $(date -Iseconds) -> $CSV ===" | tee -a "$LOG"

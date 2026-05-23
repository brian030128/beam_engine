#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_mlca_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/mlca_70b_%j.out
#SBATCH --error=slurm/logs/mlca_70b_%j.err

# Backfill mlca's 70B-fp8 results into the final_paper cells. mlca DOES
# support fp8 KV once MlcaBackend.plan is given the page-table store dtype
# (was hardcoded bf16) — fixed in baselines/mlca.py. Runs mlca-only on the
# settled configs and merges its rows into the canonical cell CSVs.
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
export BE_WORKSPACE_MB=1024
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3
export BE_HOTPOTQA_MULTI_Q_PATH=data/hotpotqa_multi_q_K128_80k.jsonl

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/final_paper/Llama-3-70B-Instruct-FP8"
LOG="$OUT_DIR/run_mlca.log"
mkdir -p "$OUT_DIR" slurm/logs
echo "=== mlca 70B-fp8 backfill started $(date -Iseconds) ===" | tee -a "$LOG"

sglang_mlca () {  # tag scenario K B canonical_csv  [env KV...]
    local tag="$1" scenario="$2" K="$3" B="$4" canon="$5"
    local tmp="$OUT_DIR/_mlca_${tag}.csv"
    echo "=== $tag: mlca $scenario K=$K B=$B ===" | tee -a "$LOG"
    rm -f "$tmp"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods mlca --no-stage2 --max-new 256 --repeat 3 --warmup \
        --scenarios "$scenario" --k-override "$K" --b-override "$B" \
        --out "$tmp" 2>&1 | tee -a "$LOG" || echo "  ($tag mlca failed)" | tee -a "$LOG"
    uv run python scripts/paper-exp/merge_method.py "$OUT_DIR/$canon" "$tmp" mlca 2>&1 | tee -a "$LOG"
}

# F1 multi_doc_qa K=128 B=1 ; F2 multi_few_shot K=128 B=4 (settled)
sglang_mlca F1 multi_doc_qa   128 1 "F1_multi_doc_qa_K128_B1_mn256.csv"
sglang_mlca F2 multi_few_shot 128 4 "F2_multi_few_shot_K128_B4_mn256.csv"

# F4 self_consistency K=16 B=1 L_p=16K
export BE_SELF_CONSISTENCY_LP=16384
sglang_mlca F4 self_consistency 16 1 "F4_self_consistency_K16_B1_Lp16384_mn256.csv"
unset BE_SELF_CONSISTENCY_LP

# F3 beam_search K=64 B=4 L_p=16000 (batched harness)
echo "=== F3: mlca beam_search K=64 B=4 L_p=16000 ===" | tee -a "$LOG"
tmp3="$OUT_DIR/_mlca_F3.csv"; rm -f "$tmp3"
torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/bench_batched.py \
    --K 64 --L_p 16000 --B 4 --max_new 256 \
    --prompts-file data/hotpotqa.jsonl --warmup --methods mlca \
    --out "$tmp3" 2>&1 | tee -a "$LOG" || echo "  (F3 mlca failed)" | tee -a "$LOG"
uv run python scripts/paper-exp/merge_method.py \
    "$OUT_DIR/F3_beam_search_K64_B4_Lp16000_mn256.csv" "$tmp3" mlca 2>&1 | tee -a "$LOG"

echo "=== mlca backfill done $(date -Iseconds) ===" | tee -a "$LOG"

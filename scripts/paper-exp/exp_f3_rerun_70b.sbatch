#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f3_rerun_70b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/f3_rerun_70b_%j.out
#SBATCH --error=slurm/logs/f3_rerun_70b_%j.err

# Re-run F3 beam_search (K=64 B=4 L_p=16000) on 70B-fp8 with:
#  - the FlashInfer-style L0 split-K fix (cascade.py), and
#  - mlca getting a real fp8 KV cache (kv_dtype fix in mlca.beam_search)
#    + a 4 GiB workspace so its cascade scratch no longer overflows.
# Each method runs in its own process (isolate any fp8 crash) and is
# merged into the canonical F3 CSV.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"; trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_WORKSPACE_MB=4096
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/final_paper/Llama-3-70B-Instruct-FP8"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run_f3_rerun.log"
CSV="$OUT_DIR/F3_beam_search_K64_B4_Lp16000_mn256.csv"
echo "=== F3 rerun (split-K + fp8 mlca) $(date -Iseconds) ===" | tee -a "$LOG"

rm -f "$CSV"
first=1
for m in paged mlca bs_kernel bs_kernel_2l1p bs_kernel_2ldt; do
    echo "--- method: $m ---" | tee -a "$LOG"
    tmp="$OUT_DIR/_f3_${m}.csv"; rm -f "$tmp"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_batched.py \
        --K 64 --L_p 16000 --B 4 --max_new 256 \
        --prompts-file data/hotpotqa.jsonl --warmup --methods "$m" \
        --out "$tmp" 2>&1 | tee -a "$LOG" || echo "  ($m failed)" | tee -a "$LOG"
    if [[ -f "$tmp" ]]; then
        if [[ $first -eq 1 ]]; then cp "$tmp" "$CSV"; first=0
        else tail -n +2 "$tmp" >> "$CSV"; fi
        rm -f "$tmp"
    fi
done
echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"

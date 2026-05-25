#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_verify_paged
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/verify_paged_%j.out
#SBATCH --error=slurm/logs/verify_paged_%j.err

# Prove the paged LCA index-dedup is bit-identical to the naive per-beam
# build (PAGED_VERIFY_LCA=1 asserts every step). Runs all four cell
# shapes on 8B; the assert raising = failure. F3 (beam_search) forks and
# diverges, the case most likely to break the shared-prefix assumption.
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
export BE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16
export BE_HOTPOTQA_MULTI_Q_PATH=data/hotpotqa_multi_q_K128_80k.jsonl
export PAGED_VERIFY_LCA=1

OUT=benchmarks/bs_kernel/results/paper-exp/_validate_paged_cache
mkdir -p "$OUT" slurm/logs
LOG="$OUT/verify.log"
: > "$LOG"

sg () {  # tag scenario K B [lp]
    echo "=== verify $1: $2 K=$3 B=$4 ===" | tee -a "$LOG"
    [[ -n "${5:-}" ]] && export BE_SELF_CONSISTENCY_LP="$5"
    torchrun --standalone --nproc_per_node=1 \
        benchmarks/bs_kernel/bench_sglang_e2e.py --methods paged --no-stage2 \
        --max-new 64 --repeat 1 --warmup --scenarios "$2" --k-override "$3" --b-override "$4" \
        --out "$OUT/_verify_$1.csv" 2>&1 | tee -a "$LOG"
}

sg F1 multi_doc_qa     128 1
sg F2 multi_few_shot   128 4
sg F4 self_consistency 16  1 16384

echo "=== verify F3: beam_search K=64 B=8 (batched, forking) ===" | tee -a "$LOG"
torchrun --standalone --nproc_per_node=1 \
    benchmarks/bs_kernel/bench_batched.py --K 64 --L_p 16000 --B 8 --max_new 64 \
    --prompts-file data/hotpotqa.jsonl --warmup --methods paged \
    --out "$OUT/_verify_F3.csv" 2>&1 | tee -a "$LOG"

if grep -qiE "LCA dedup mismatch|AssertionError|Traceback" "$LOG"; then
    echo "VERIFY: FAILED (see $LOG)" | tee -a "$LOG"
else
    echo "VERIFY: PASS — paged LCA dedup is bit-identical on all cells" | tee -a "$LOG"
fi

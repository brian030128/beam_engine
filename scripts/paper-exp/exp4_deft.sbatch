#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp4_deft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/paper_exp4_deft_%j.out
#SBATCH --error=slurm/logs/paper_exp4_deft_%j.err

# Companion to exp4_long_decode.sbatch — runs the DeFT baseline only and
# appends a row to the existing merged.csv. DeFT is ~20-30x slower per
# step than the other four methods, so it ships as a separate sbatch
# rather than blocking the main exp4 job's 40-min slot.
#
# Cell: Llama-3.2-1B, K=32, L_p=8192, B=16, max_new=2048
#   ~150-200 ms/decode-step × 2047 steps × 2 (warmup + measured)
#   ≈ 12 min wallclock, fits the 40-min cap with margin.
#
# Output:
#   benchmarks/bs_kernel/results/paper-exp/exp4_long_decode/deft.csv  (raw)
#   appended into the same dir's merged.csv (header skipped).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp4_long_decode"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp4_deft.log"

{
    echo "=== exp4 DeFT-only run $(date -Iseconds) ==="
    echo "model:    $BE_MODEL"
    echo "prompts:  data/gov_report.jsonl"
    echo "cell:     L_p=8192, K=32, B=16, max_new=2048"
    echo "method:   deft"
    echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 32 --L_p 8192 --B 16 --max_new 2048 \
    --prompts-file data/gov_report.jsonl \
    --max_pages 80000 \
    --warmup \
    --methods deft \
    --out "$OUT_DIR/deft.csv" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [[ $rc -ne 0 ]]; then
    echo "  >>> deft: python exited with $rc" | tee -a "$LOG"
fi

# Append into the existing merged.csv (skip the deft.csv header row).
MERGED="$OUT_DIR/merged.csv"
if [[ -f "$OUT_DIR/deft.csv" && -f "$MERGED" ]]; then
    echo "=== appending deft.csv into $MERGED ===" | tee -a "$LOG"
    tail -n +2 "$OUT_DIR/deft.csv" >> "$MERGED"
    echo "merged size: $(wc -l < "$MERGED") lines" | tee -a "$LOG"
elif [[ ! -f "$MERGED" ]]; then
    echo "WARN: $MERGED does not exist (run exp4_long_decode.sbatch first); leaving deft.csv standalone." | tee -a "$LOG"
fi

{
    echo
    echo "=== updated merged results ==="
    [[ -f "$MERGED" ]] && cat "$MERGED"
    echo
    echo "Done. Results: $OUT_DIR"
} | tee -a "$LOG"

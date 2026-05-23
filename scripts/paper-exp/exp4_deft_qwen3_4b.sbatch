#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_exp4_dft_q3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/paper_exp4_deft_qwen3_%j.out
#SBATCH --error=slurm/logs/paper_exp4_deft_qwen3_%j.err

# Qwen3-4B DeFT-only long-decode — companion to exp4_long_decode_qwen3_4b.sbatch.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL="Qwen/Qwen3-4B"

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp4_long_decode_qwen3_4b"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp4_deft_qwen3.log"

{
    echo "=== exp4 DeFT Qwen3-4B run $(date -Iseconds) ==="
    echo "model:    $BE_MODEL"
    echo "HF_HOME:  $HF_HOME"
    echo "prompts:  data/gov_report.jsonl"
    echo "cell:     L_p=8192, K=32, B=16, max_new=2048"
    echo "method:   deft"
    echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 32 --L_p 8192 --B 4 --max_new 2048 \
    --prompts-file data/gov_report.jsonl \
    --max_pages 20000 \
    --warmup \
    --methods deft \
    --out "$OUT_DIR/deft.csv" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [[ $rc -ne 0 ]]; then
    echo "  >>> deft: python exited with $rc" | tee -a "$LOG"
fi

MERGED="$OUT_DIR/merged.csv"
if [[ -f "$OUT_DIR/deft.csv" && -f "$MERGED" ]]; then
    echo "=== appending deft.csv into $MERGED ===" | tee -a "$LOG"
    tail -n +2 "$OUT_DIR/deft.csv" >> "$MERGED"
    echo "merged size: $(wc -l < "$MERGED") lines" | tee -a "$LOG"
elif [[ ! -f "$MERGED" ]]; then
    echo "WARN: $MERGED does not exist (run exp4_long_decode_qwen3_4b.sbatch first); leaving deft.csv standalone." | tee -a "$LOG"
fi

{
    echo
    echo "=== updated merged results ==="
    [[ -f "$MERGED" ]] && cat "$MERGED"
    echo
    echo "Done. Results: $OUT_DIR"
} | tee -a "$LOG"

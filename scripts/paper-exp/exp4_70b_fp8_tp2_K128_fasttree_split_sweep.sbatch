#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_ft_split_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/paper_ft_split_sweep_%j.out
#SBATCH --error=slurm/logs/paper_ft_split_sweep_%j.err

# FastTree split-K sweep on RedHatAI/Meta-Llama-3-70B-Instruct-FP8
# TP=2, K=128, L_p=8192, B=2, max_new=768 — the same cell where
# fasttree was 2.93× behind bs_kernel at default (1024, 128) splits.
#
# Only KV_SPLIT_SIZES is varied (via FT_KV_SPLIT_SIZES env var);
# para_threshs1 and para_threshs2 are kept at FastTree's shipped
# defaults (132,528) / (132,132) so we isolate the split-K knob.
#
# Hypothesis: 70B-fp8 TP=2 has per-rank num_kv_heads=4 (vs 8 at
# single-GPU on the smaller models), so the kernel launches fewer
# parallel CTAs per step. The shipped default fallback split of 128
# tokens means each CTA does only 128*4*128 bytes = 64 KiB of K/V
# work — at H100 HBM bandwidth this is tiny per launch. Bigger splits
# should amortize per-launch cost.
#
# Sweep:
#   (1024,  128)  — shipped default (control; matches prior K=128 run)
#   (2048,  256)  — 2x bigger default + 2x bigger fallback
#   (4096,  512)  — 4x bigger
#   (8192, 1024)  — splits = full L_p / partial, max parallelism floor
#
# No warmup per method to keep the wall time tight; each method is
# already a single-pass measurement with 767 decode steps.
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp4_70b_fp8_tp2_K128_ft_split_sweep/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp4_70b_fp8_tp2_K128_ft_split_sweep"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/ft_split_sweep.log"

{
    echo "=== fasttree KV_SPLIT_SIZES sweep on 70B-fp8 TP=2 ==="
    echo "started:  $(date -Iseconds)"
    echo "model:    $BE_MODEL"
    echo "cell:     K=128, L_p=8192, B=2, max_new=768"
    echo "control:  KV_SPLIT_SIZES=(1024,128) was 139,740 ms (job 206782)"
    echo
} | tee "$LOG"

# Tag → "default,fallback" pairs.
declare -A SPLITS=(
    [1024_128]="1024,128"
    [2048_256]="2048,256"
    [4096_512]="4096,512"
    [8192_1024]="8192,1024"
)

for tag in 1024_128 2048_256 4096_512 8192_1024; do
    split="${SPLITS[$tag]}"
    echo "=== KV_SPLIT_SIZES = ($split) ===" | tee -a "$LOG"
    FT_KV_SPLIT_SIZES="$split" \
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_batched.py \
        --K 128 --L_p 8192 --B 2 --max_new 768 \
        --prompts-file data/gov_report.jsonl \
        --methods fasttree \
        --out "$OUT_DIR/fasttree_${tag}.csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (split $split failed; continuing)" | tee -a "$LOG"
    echo | tee -a "$LOG"
done

MERGED="$OUT_DIR/merged.csv"
echo "=== merging CSVs to $MERGED ===" | tee -a "$LOG"
first=1
for tag in 1024_128 2048_256 4096_512 8192_1024; do
    f="$OUT_DIR/fasttree_${tag}.csv"
    [[ -f "$f" ]] || { echo "WARN: $f missing" | tee -a "$LOG"; continue; }
    if [[ $first -eq 1 ]]; then
        # Rewrite method column to include the split tag for readability.
        head -1 "$f" | sed 's/^method/method_split/' > "$MERGED"
        tail -n +2 "$f" | sed "s/^fasttree/fasttree_${tag}/" >> "$MERGED"
        first=0
    else
        tail -n +2 "$f" | sed "s/^fasttree/fasttree_${tag}/" >> "$MERGED"
    fi
done

{
    echo
    echo "=== merged results (sorted by decode_total_ms) ==="
    if [[ -f "$MERGED" ]]; then
        # Pretty-print: header + rows sorted by decode_total_ms (col 7).
        head -1 "$MERGED"
        tail -n +2 "$MERGED" | sort -t, -k7 -n
    else
        echo "(no CSVs produced)"
    fi
    echo
    echo "Done. Results: $OUT_DIR"
} | tee -a "$LOG"

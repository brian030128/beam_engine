#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_ft_split_smaller
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/paper_ft_split_smaller_%j.out
#SBATCH --error=slurm/logs/paper_ft_split_smaller_%j.err

# Continuation of exp4_70b_fp8_tp2_K128_fasttree_split_sweep —
# previous sweep showed bigger splits monotonically worse. Try the
# other direction: smaller splits → more CTAs → more parallelism on
# H100's 132 SMs.
#
# Sweep (default, fallback):
#   (512,  64)
#   (256,  32)
#   (128,  16)
#
# para_threshs* untouched (per the user's "only split-K" constraint).
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp4_70b_fp8_tp2_K128_ft_split_smaller/

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

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp4_70b_fp8_tp2_K128_ft_split_smaller"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/ft_split_smaller.log"

{
    echo "=== fasttree KV_SPLIT_SIZES smaller-sweep on 70B-fp8 TP=2 ==="
    echo "started:  $(date -Iseconds)"
    echo "cell:     K=128, L_p=8192, B=2, max_new=768"
    echo "control:  (1024,128) = 140,849 ms; (2048,256) = 140,687 ms"
    echo "tested:   (4096,512) = 203,786 ms; (8192,1024) = 264,310 ms"
    echo
} | tee "$LOG"

declare -A SPLITS=(
    [512_64]="512,64"
    [256_32]="256,32"
    [128_16]="128,16"
)

for tag in 512_64 256_32 128_16; do
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
for tag in 512_64 256_32 128_16; do
    f="$OUT_DIR/fasttree_${tag}.csv"
    [[ -f "$f" ]] || { echo "WARN: $f missing" | tee -a "$LOG"; continue; }
    if [[ $first -eq 1 ]]; then
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
        head -1 "$MERGED"
        tail -n +2 "$MERGED" | sort -t, -k7 -n
    fi
    echo
    echo "Done. Results: $OUT_DIR"
} | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=2l1p_old
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:35:00
#SBATCH --output=slurm/logs/bench_2l1p_old_baseline_%j.out
#SBATCH --error=slurm/logs/bench_2l1p_old_baseline_%j.err

# Re-measure OLD bs_kernel_2l1p (multi-launch, no single_launch=True) at
# the exact cells of the new 2l1p_single vs 2ldt sweep, so we can do an
# apples-to-apples 3-way comparison (old 2l1p / new 2l1p_single / 2ldt).
# Two argv: model_alias (1b|8b).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf

MODEL_ALIAS="${1:-1b}"
if [[ "$MODEL_ALIAS" == "1b" ]]; then
    export BE_MODEL=meta-llama/Llama-3.2-1B-Instruct
    cells=("16 1" "16 4" "16 32" "32 1" "32 4" "32 32" "64 1" "64 4" "64 32")
else
    export BE_MODEL=meta-llama/Llama-3.1-8B
    cells=("16 1" "16 4" "32 1" "32 4" "64 1" "64 4")
fi

mkdir -p slurm/logs
TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/2l1p_old_baseline_${MODEL_ALIAS}-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== old bs_kernel_2l1p baseline | ${MODEL_ALIAS} | started $(date -Iseconds) ==="
    echo "model:  $BE_MODEL"
    echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
} | tee "$LOG"

for cell in "${cells[@]}"; do
    K=$(echo "$cell" | awk '{print $1}')
    B=$(echo "$cell" | awk '{print $2}')
    MAX_PAGES=$(( B * (8192 + K * 256) / 16 + 1024 ))
    tag="bs_kernel_2l1p_K${K}_B${B}"
    csv="$RESULTS_DIR/${tag}.csv"
    echo "=== ${tag} | max_pages=${MAX_PAGES} ===" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K "$K" --L_p 8192 --B "$B" --max_new 256 \
        --max_pages "$MAX_PAGES" --warmup \
        --methods bs_kernel_2l1p \
        --out "$csv" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    [[ $rc -ne 0 ]] && echo "  >>> ${tag}: python exited with $rc" | tee -a "$LOG"
    echo | tee -a "$LOG"
done

MERGED="$RESULTS_DIR/merged.csv"
first=1
for f in "$RESULTS_DIR"/*.csv; do
    [[ "$f" == "$MERGED" ]] && continue
    [[ -f "$f" ]] || continue
    if [[ $first -eq 1 ]]; then cat "$f" > "$MERGED"; first=0
    else tail -n +2 "$f" >> "$MERGED"; fi
done
{ echo "=== merged ==="; cat "$MERGED"; echo "Done. Results: $RESULTS_DIR"; } | tee -a "$LOG"

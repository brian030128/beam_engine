#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=2l1ps_mn2k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/bench_2l1p_single_vs_2ldt_mn2048_%j.out
#SBATCH --error=slurm/logs/bench_2l1p_single_vs_2ldt_mn2048_%j.err

# Forced 2l1p_single vs 2ldt at the SAME (K, L_p, max_new) as old paper-exp
# data, so we can compare apples-to-apples against existing
# bs_kernel_2l1p (multi-launch) numbers without re-measuring the old runs.
# Grid: K=32, L_p=8192, max_new=2048, B ∈ {1, 2, 4, 16}.
# Arg1: model alias (1b | 8b).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf

MODEL_ALIAS="${1:-8b}"
if [[ "$MODEL_ALIAS" == "1b" ]]; then
    export BE_MODEL=meta-llama/Llama-3.2-1B-Instruct
else
    export BE_MODEL=meta-llama/Llama-3.1-8B
fi

mkdir -p slurm/logs
TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/2l1p_single_vs_2ldt_${MODEL_ALIAS}_mn2048-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== ${MODEL_ALIAS} max_new=2048 | started $(date -Iseconds) ==="
    echo "grid:   K=32, L_p=8192, max_new=2048, B∈{1,2,4,16}"
    echo "model:  $BE_MODEL"
    echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "flashinfer diff (single_launch must be live):"
    git -C 3rdparty/flashinfer diff --stat 2>&1 | sed 's/^/  /'
} | tee "$LOG"

methods=(bs_kernel_2l1p_single bs_kernel_2ldt)
Bs=(1 2 4 16)

for m in "${methods[@]}"; do
    for B in "${Bs[@]}"; do
        K=32
        # max_pages worst case = B*(L_p + K*max_new)/16 + headroom.
        MAX_PAGES=$(( B * (8192 + K * 2048) / 16 + 1024 ))
        tag="${m}_K${K}_B${B}"
        csv="$RESULTS_DIR/${tag}.csv"
        echo "=== ${tag} | max_pages=${MAX_PAGES} ===" | tee -a "$LOG"
        uv run python benchmarks/bs_kernel/bench_batched.py \
            --K "$K" --L_p 8192 --B "$B" --max_new 2048 \
            --max_pages "$MAX_PAGES" --warmup \
            --methods "$m" \
            --out "$csv" 2>&1 | tee -a "$LOG"
        rc=${PIPESTATUS[0]}
        [[ $rc -ne 0 ]] && echo "  >>> ${tag}: python exited with $rc" | tee -a "$LOG"
        echo | tee -a "$LOG"
    done
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

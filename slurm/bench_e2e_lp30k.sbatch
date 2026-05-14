#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=e2e_lp30k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_e2e_lp30k_%j.out
#SBATCH --error=slurm/logs/bench_e2e_lp30k_%j.err

# E2E beam-search sweep: L_p=30000, B in {16,32}, K in {16,32}, max_new=128
# Methods: paged, mlca, fasttree, bs_kernel
# Models : Llama-3.2-1B, Llama-3.1-8B
# OOMs are tolerated (continue on failure).

set -uo pipefail   # no -e: a single method's OOM must not abort the loop

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/e2e_lp30k-${TS}"
mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/bench.log"

{
    echo "=== bench_e2e_lp30k started $(date -Iseconds) ==="
    echo "config: L_p=30000, max_new=128, methods=(paged mlca fasttree bs_kernel)"
    echo "cells:  (B,K) in (16,16) (16,32) (32,16) (32,32)"
    echo "models: Llama-3.2-1B, Llama-3.1-8B"
    echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

methods=(paged mlca fasttree bs_kernel)

# Per-cell max_num_pages caps (sized for shared-prompt usage + headroom):
#   prefill_shared = 30000/16 = 1875 pages
#   decode = B*K * (128/16 + 2) = B*K*10 pages
#   + 512 headroom
declare -A MAXP
MAXP[16,16]=6000     # 1875 + 2560 + 512  -> round up
MAXP[16,32]=10000    # 1875 + 5120 + 512
MAXP[32,16]=10000    # 1875 + 5120 + 512
MAXP[32,32]=16000    # 1875 + 10240 + 512

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    export BE_MODEL="meta-llama/${model_tag}"
    echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"
    for B in 16 32; do
        for K in 16 32; do
            cap=${MAXP[$B,$K]}
            for m in "${methods[@]}"; do
                tag="${model_tag}-B${B}-K${K}-${m}"
                csv="$RESULTS_DIR/${tag}.csv"
                echo "=== ${tag}  max_pages=${cap} ===" | tee -a "$LOG"
                # Each (model,cell,method) is a fresh python process so
                # KV slab from one method cannot starve the next.
                uv run python benchmarks/bs_kernel/bench_batched.py \
                    --K "$K" \
                    --L_p 30000 \
                    --B "$B" \
                    --max_new 128 \
                    --max_pages "$cap" \
                    --methods "$m" \
                    --out "$csv" \
                    2>&1 | tee -a "$LOG"
                rc=${PIPESTATUS[0]}
                if [[ $rc -ne 0 ]]; then
                    echo "  >>> ${tag}: python exited with $rc (likely OOM)" | tee -a "$LOG"
                fi
                echo | tee -a "$LOG"
            done
        done
    done
done

# Merge per-(model,cell,method) CSVs.
MERGED="$RESULTS_DIR/merged.csv"
echo "=== merging CSVs to $MERGED ===" | tee -a "$LOG"
first=1
for f in "$RESULTS_DIR"/*.csv; do
    [[ -f "$f" ]] || continue
    if [[ $first -eq 1 ]]; then
        # add a "model" column at the front so each row records which model
        head -n 1 "$f" | awk '{print "model,"$0}' > "$MERGED"
        first=0
    fi
    base=$(basename "$f" .csv)
    model_field=$(echo "$base" | cut -d- -f1-3)  # e.g. Llama-3.2-1B
    tail -n +2 "$f" | awk -v m="$model_field" '{print m","$0}' >> "$MERGED"
done

{
    echo
    echo "=== merged results ==="
    cat "$MERGED"
    echo
    echo "Done. Results: $RESULTS_DIR"
} | tee -a "$LOG"

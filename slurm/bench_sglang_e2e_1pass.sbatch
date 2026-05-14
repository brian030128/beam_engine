#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=sg_e2e_1p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_sglang_e2e_1pass_%j.out
#SBATCH --error=slurm/logs/bench_sglang_e2e_1pass_%j.err

# Single-pass SGLang E2E across 4 scenarios x 4 methods x 2 models.
# OOMs / errors per (scenario, method) are caught inside the python
# script; the loop here only switches models.

set -uo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/sglang_e2e_1pass-${TS}"
mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/bench.log"

{
    echo "=== bench_sglang_e2e_1pass started $(date -Iseconds) ==="
    echo "scenarios: multi_level_system multi_few_shot multi_chain_reasoning multi_document"
    echo "methods:   bs_kernel fasttree paged mlca"
    echo "max_new:   256"
    echo "repeat:    1   (no warmup)"
    echo "models:    Llama-3.2-1B, Llama-3.1-8B"
    echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    export BE_MODEL="meta-llama/${model_tag}"
    csv="$RESULTS_DIR/${model_tag}.csv"
    echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_level_system multi_few_shot \
                    multi_chain_reasoning multi_document \
        --methods bs_kernel fasttree paged mlca \
        --max-new 256 \
        --repeat 1 \
        --out "$csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

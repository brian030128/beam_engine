#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=mlca_chain_rerun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:59:00
#SBATCH --output=slurm/logs/mlca_chain_rerun_%j.out
#SBATCH --error=slurm/logs/mlca_chain_rerun_%j.err

# Re-run MLCA on multi_chain_reasoning for both 1B and 8B after the
# FlashInfer-path illegal-memory-access fix. Mirrors the MLCA sub-call
# in scripts/paper-exp/exp1a_end_to_end_timing.sbatch.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/mlca_chain_rerun.log"

{
    echo "=== mlca chain_reasoning rerun started $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    if [[ "$model_tag" == "Llama-3.2-1B" ]]; then
        chain_B=32
    else
        chain_B=8
    fi
    csv="$OUT_DIR/${model_tag}_chain_reasoning_mlca.csv"
    echo "############ ${model_tag}: multi_chain_reasoning B=${chain_B} K=64 ############" | tee -a "$LOG"
    BE_MODEL="meta-llama/${model_tag}" uv run python \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods mlca --paper-exact \
        --max-new 256 --repeat 1 --warmup \
        --scenarios multi_chain_reasoning --k-override 64 --b-override "$chain_B" \
        --out "$csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (mlca ${model_tag} chain_reasoning failed)" | tee -a "$LOG"
done

echo "Done. Results: $OUT_DIR" | tee -a "$LOG"

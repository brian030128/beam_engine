#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=diag_cell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/diag_cell_%j.out
#SBATCH --error=slurm/logs/diag_cell_%j.err

# Per-step bs_kernel cost-model diagnostic at the failing cell:
# K=64, L_p=8192, B=32, max_new=256, distinct prompts.
#
# Each mode runs in its own python invocation so KV-page memory from one
# mode cannot leak into the next.

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/diagnose_cell-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/diag.log"

modes=(auto per_beam 2l1p 2l2p 3l1p 3l2p)

{
    echo "=== diagnose_cell started $(date -Iseconds) ==="
    echo "K=64 L_p=8192 B=32 max_new=256 distinct=True"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

for m in "${modes[@]}"; do
    echo "=== mode: $m ===" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/diagnose_cell.py run \
        --mode "$m" \
        --out "$RESULTS_DIR/${m}.json" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "=== merging into per-step CSV ===" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/diagnose_cell.py merge \
    --indir "$RESULTS_DIR" \
    --out "$RESULTS_DIR/per_step.csv" \
    2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=prof_disp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/prof_disp_%j.out
#SBATCH --error=slurm/logs/prof_disp_%j.err

# Per-phase profile of the bs_kernel decode loop at the failing cell.
# Runs 2l1p (smallest dispatch path: fused 2-level) + 3l2p (full path
# with depth=3 layout work and non-fused 5-launch dispatch). Compares
# per-phase Python wall-time so we know exactly where the 10s gap to
# adaptive_pool comes from.

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/profile_dispatch-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/profile.log"

modes=(2l1p 3l2p)

{
    echo "=== profile_dispatch started $(date -Iseconds) ==="
    echo "K=64 L_p=8192 B=32 max_new=256 distinct=True"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

for m in "${modes[@]}"; do
    echo "=== mode: $m (run) ===" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/diagnose_cell.py run \
        --mode "$m" \
        --out "$RESULTS_DIR/${m}.json" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

for m in "${modes[@]}"; do
    echo "=== mode: $m (phase_summary) ===" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/diagnose_cell.py phase_summary \
        --input "$RESULTS_DIR/${m}.json" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

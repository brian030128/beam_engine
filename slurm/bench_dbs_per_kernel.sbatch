#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=dbs_per_kernel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/dbs_per_kernel_%j.out
#SBATCH --error=slurm/logs/dbs_per_kernel_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3

source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/dbs-per-kernel-${TS}"
mkdir -p "$RESULTS_DIR"

echo "=== Sweep: 6 base kernels + 6 DBS variants on the quick grid ==="
for m in paged tree fasttree mlca adaptive_pool bs_kernel \
         dbs_paged dbs_tree dbs_fasttree dbs_mlca dbs_adaptive_pool dbs_bs_kernel; do
    echo "--- $m ---"
    uv run python benchmarks/bs_kernel/sweep.py --quick --methods "$m" \
        --out "$RESULTS_DIR/sweep-$m.csv" 2>&1 | tee -a "$RESULTS_DIR/sweep.log"
done

echo
echo "=== Demo: single-prompt headline (K=16, L_p=8k) — all 12 methods ==="
uv run python benchmarks/bs_kernel/demo_e2e.py \
    --K 16 --L_p 8192 --max_new 32 \
    2>&1 | tee "$RESULTS_DIR/demo-K16-Lp8k.log"

echo
echo "Done. Results: $RESULTS_DIR"

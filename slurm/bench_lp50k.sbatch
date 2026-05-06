#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_lp50k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/bench_lp50k_%j.out
#SBATCH --error=slurm/logs/bench_lp50k_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/lp50k-${TS}"
mkdir -p "$RESULTS_DIR"

# 6 base kernels at K ∈ {16,32,64} × L_p=51200 (50k, page_size=16 aligned).
for m in paged tree fasttree mlca adaptive_pool bs_kernel; do
    echo "--- $m ---"
    uv run python benchmarks/bs_kernel/sweep.py \
        --K 16 32 64 --L_p 51200 --max_new 16 \
        --methods "$m" --out "$RESULTS_DIR/sweep-$m.csv" \
        2>&1 | tee -a "$RESULTS_DIR/sweep.log"
done

echo
echo "Done. Results: $RESULTS_DIR"

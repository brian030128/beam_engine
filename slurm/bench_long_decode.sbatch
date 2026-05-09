#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bench_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_long_decode_%j.out
#SBATCH --error=slurm/logs/bench_long_decode_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bench_long_decode-${TS}"
mkdir -p "$RESULTS_DIR"

# K {16, 64} x L_p {8192, 50000} x B {1, 16, 32} x max_new=512
# OOM-prone cells (K=64+B>=16, or L_p=50k+B>=16) print FAILED and the row
# is skipped in the CSV; survivors are reported.
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 16 64 \
    --L_p 8192 50000 \
    --B 1 16 32 \
    --max_new 512 \
    --methods paged tree fasttree mlca adaptive_pool bs_kernel \
    --out "$RESULTS_DIR/bench_long_decode.csv" \
    2>&1 | tee "$RESULTS_DIR/bench_long_decode.log"

echo
echo "Done. Results: $RESULTS_DIR"

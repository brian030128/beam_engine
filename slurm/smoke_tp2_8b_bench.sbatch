#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=tp2_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/smoke_tp2_8b_bench_%j.out
#SBATCH --error=slurm/logs/smoke_tp2_8b_bench_%j.err

# End-to-end smoke for the TP=2 bench wiring before submitting full
# paper-exp jobs. Single cell × 2 methods × short max_new.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.1-8B

OUT=/tmp/smoke_tp2_8b_bench.csv

echo "=== bench_sglang_e2e.py TP=2 smoke ==="
torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_level_system \
    --methods bs_kernel paged \
    --paper-exact --k-override 64 \
    --max-new 32 --repeat 1 --warmup \
    --out "$OUT"
echo "  -> $OUT"
ls -l "$OUT"
echo

echo "=== bench_batched.py TP=2 smoke ==="
OUT2=/tmp/smoke_tp2_8b_batched.csv
torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/bench_batched.py \
    --K 16 --L_p 2048 --B 4 --max_new 16 \
    --methods bs_kernel paged \
    --out "$OUT2"
echo "  -> $OUT2"
ls -l "$OUT2"

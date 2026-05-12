#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=plan_k16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/profile_plan_k16_lp30k_%j.out
#SBATCH --error=slurm/logs/profile_plan_k16_lp30k_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Trace sub-phases inside plan_decode_step for both methods.
export BS_KERNEL_TRACE_PLAN=1
export FT_TRACE_PLAN=1

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/profile_plan_k16_lp30k-${TS}"
mkdir -p "$RESULTS_DIR"

uv run python benchmarks/bs_kernel/profile_fasttree_vs_bs.py \
    --K 16 --L_p 30000 --B 8 --max_new 256 \
    --methods fasttree bs_kernel \
    2>&1 | tee "$RESULTS_DIR/result.log"

echo
echo "Done. Results: $RESULTS_DIR"

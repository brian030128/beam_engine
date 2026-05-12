#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=plan_cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/test_plan_cache_k16_%j.out
#SBATCH --error=slurm/logs/test_plan_cache_k16_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo " STEP 1: correctness smoke (bs_kernel == paged, K=16/B=4)"
echo "============================================================"
uv run python benchmarks/bs_kernel/smoke_plan_cache.py
echo

echo "============================================================"
echo " STEP 2: plan-trace at K=16 across B values"
echo "============================================================"
export BS_KERNEL_TRACE_PLAN=1
for B in 1 4 8 16 32; do
    echo "------------------------------------------------------------"
    echo " K=16 L_p=8192 B=${B}  (with plan-state cache)"
    echo "------------------------------------------------------------"
    uv run python benchmarks/bs_kernel/profile_fasttree_vs_bs.py \
        --K 16 --L_p 8192 --B "${B}" --max_new 256 \
        --methods bs_kernel fasttree
    echo
done

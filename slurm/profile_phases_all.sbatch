#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=phases_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/profile_phases_all_%j.out
#SBATCH --error=slurm/logs/profile_phases_all_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Cross-method phase breakdown for paged / mlca / fasttree / bs_kernel
# at K=64, L_p=30000, B=16, max_new=256. bs_kernel also dumps its
# per-step cost-model pick histogram.
#
# B reduced from 32 -> 16 because page_driver prefill packs all B*L_p
# tokens into a single forward; at L_p=30000 the FFN gate_up_proj
# allocation for B=32 is ~30 GiB which OOMs the H100. B=16 keeps that
# under ~15 GiB while still exercising cross-prompt batching.
uv run python benchmarks/bs_kernel/profile_phases_all.py \
    --K 64 \
    --L_p 30000 \
    --B 16 \
    --max_new 256 \
    --methods paged mlca fasttree bs_kernel

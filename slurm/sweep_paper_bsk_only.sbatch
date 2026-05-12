#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=sweep_bsk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/sweep_paper_bsk_only_%j.out
#SBATCH --error=slurm/logs/sweep_paper_bsk_only_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
OUT_DIR="benchmarks/bs_kernel/results/sweep_paper_bsk-${TS}"

# bs_kernel + dbs_bs_kernel only — other methods unchanged, will be
# merged from the prior sweep (sweep_paper-20260512-161859).
uv run python benchmarks/bs_kernel/sweep_paper.py \
    --K 4 16 32 64 \
    --L_p 8192 30000 \
    --B 1 4 8 16 32 \
    --max_new 256 \
    --methods bs_kernel dbs_bs_kernel \
    --out_dir "${OUT_DIR}"

echo
echo "Results written to: ${OUT_DIR}"
ls -la "${OUT_DIR}/"

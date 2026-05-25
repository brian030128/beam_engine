#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=smk_unop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/smoke_unop_fasttree_%j.out
#SBATCH --error=slurm/logs/smoke_unop_fasttree_%j.err

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B

OUT=/home/u4320956/beam_engine-v3/slurm/logs/smoke_unop_fasttree_${SLURM_JOB_ID}.csv

echo "=== smoke: unop_fasttree vs fasttree (1B, K=8, B=2, L_p=512, max_new=16) ==="
uv run python benchmarks/bs_kernel/bench_batched.py \
    --K 8 --L_p 512 --B 2 --max_new 16 \
    --warmup \
    --methods fasttree unop_fasttree \
    --out "$OUT"

echo "=== CSV ==="
cat "$OUT"

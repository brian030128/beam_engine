#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=smk2stg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/smoke_two_stage_%j.out
#SBATCH --error=slurm/logs/smoke_two_stage_%j.err

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== smoke: paper-exact multi_chain_reasoning two-stage (1B, fasttree only, max_new=32) ==="
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_chain_reasoning \
    --methods fasttree \
    --paper-exact \
    --max-new 32 --repeat 1 \
    --out /tmp/smoke_two_stage.csv
echo "=== CSV ==="
cat /tmp/smoke_two_stage.csv

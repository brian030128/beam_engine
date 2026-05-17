#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=mlca_plan_brk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/mlca_plan_brk_%j.out
#SBATCH --error=slurm/logs/mlca_plan_brk_%j.err

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

LOG="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing/mlca_plan_breakdown.log"
{
    echo "=== mlca plan-phase breakdown $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

for cell in 1B-fewshot 1B-chain 1B-levelsys 8B-fewshot 8B-chain 8B-levelsys; do
    echo "==================================================" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_mlca_plan_breakdown.py \
        --cell "$cell" --iters 50 2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "Done." | tee -a "$LOG"

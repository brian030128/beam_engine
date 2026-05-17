#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=paged_trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/paged_trace_%j.out
#SBATCH --error=slurm/logs/paged_trace_%j.err

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PAGED_TRACE_PLAN=1
export BE_MODEL=meta-llama/Llama-3.2-1B

LOG="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing/paged_trace.log"
{ echo "=== paged per-phase trace $(date -Iseconds) ==="; } | tee "$LOG"

uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods paged --paper-exact \
    --max-new 256 --repeat 1 --warmup \
    --scenarios multi_few_shot --k-override 64 --b-override 32 \
    --out /tmp/_paged_trace.csv 2>&1 | tee -a "$LOG"
echo "Done." | tee -a "$LOG"

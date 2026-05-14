#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=ft_mls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/profile_ft_mls_%j.out
#SBATCH --error=slurm/logs/profile_ft_mls_%j.err

# Trace + perf check for fasttree on multi_level_system (1B).
# 1. FT_TRACE_PLAN=1 trace to confirm meta_ms dropped.
# 2. Plain run to capture decode_total / p50/step.
# 3. Cache-disabled run for A/B reference.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [1/3] FT_TRACE_PLAN trace (cache ON) ==="
FT_TRACE_PLAN=1 uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_level_system --methods fasttree \
    --max-new 64 --repeat 1

echo
echo "=== [2/3] full multi_level_system, cache ON ==="
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_level_system --methods fasttree paged mlca bs_kernel \
    --max-new 256 --repeat 1

echo
echo "=== [3/3] full multi_level_system, cache OFF (A/B) ==="
BE_FT_PLAN_CACHE=0 uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_level_system --methods fasttree \
    --max-new 256 --repeat 1

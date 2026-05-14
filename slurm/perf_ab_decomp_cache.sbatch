#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bsk_perf_ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/perf_ab_decomp_cache_%j.out
#SBATCH --error=slurm/logs/perf_ab_decomp_cache_%j.err

set -euo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1

echo "###############################################"
echo "# A: decomp cache ENABLED (default)"
echo "###############################################"
unset BS_KERNEL_DISABLE_DECOMP_CACHE
for SC in multi_chain_reasoning multi_document multi_level_system; do
    echo
    echo "===== scenario: $SC ====="
    uv run python benchmarks/bs_kernel/profile_sglang_e2e.py \
        --scenario $SC \
        --methods bs_kernel \
        --max-new 256
done

echo
echo "###############################################"
echo "# B: decomp cache DISABLED (baseline)"
echo "###############################################"
export BS_KERNEL_DISABLE_DECOMP_CACHE=1
for SC in multi_chain_reasoning multi_document multi_level_system; do
    echo
    echo "===== scenario: $SC ====="
    uv run python benchmarks/bs_kernel/profile_sglang_e2e.py \
        --scenario $SC \
        --methods bs_kernel \
        --max-new 256
done

echo
echo "###############################################"
echo "# C: full E2E (cache enabled) for ranking update"
echo "###############################################"
unset BS_KERNEL_DISABLE_DECOMP_CACHE
unset BS_KERNEL_TRACE_PLAN

uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_level_system multi_few_shot multi_chain_reasoning multi_document \
    --methods bs_kernel fasttree paged mlca \
    --max-new 256 \
    --repeat 3 \
    --warmup

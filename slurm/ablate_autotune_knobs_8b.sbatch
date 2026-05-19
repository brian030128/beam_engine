#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=ablate_knob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/ablate_autotune_knobs_8b_%j.out
#SBATCH --error=slurm/logs/ablate_autotune_knobs_8b_%j.err

# Ablation over the 4 autotuned cost-model knobs:
#   share_extra_us, dual_pool_extra_us,
#   dec_tail_extra_us, dec_tail_per_tail_page_us
#
# Question: are all four necessary on H100 + Llama-3.1-8B?
# Methodology: measure the autotune grid once, then grid-search the
# regret subject to each subset of knobs being clamped to 0.
#
# This is a READ-ONLY experiment — it does NOT touch the coefficient
# cache. Safe to run alongside ongoing benches.

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
echo

uv run python benchmarks/bs_kernel/ablate_autotune_knobs.py \
    --model meta-llama/Llama-3.1-8B \
    --max-new 256

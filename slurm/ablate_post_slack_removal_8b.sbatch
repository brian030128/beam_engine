#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=ab_post_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/ablate_post_slack_removal_8b_%j.out
#SBATCH --error=slurm/logs/ablate_post_slack_removal_8b_%j.err

# Step 2 + 3 regression: share_extra_us / dec_tail_extra_us removed,
# DEC_TAIL tail wave-quantized. The picker has no autotuned slack —
# pure closed-form. We want to see the regret on the AUTOTUNE_GRID
# vs the previous baseline (which had share=100 fitting on this model).

set -euo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python benchmarks/bs_kernel/ablate_autotune_knobs.py \
    --model meta-llama/Llama-3.1-8B \
    --max-new 256

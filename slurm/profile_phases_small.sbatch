#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=profile_small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/profile_phases_small_%j.out
#SBATCH --error=slurm/logs/profile_phases_small_%j.err

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Same K=4/B=4/L_p=8192 as the small bench, max_new=128 to keep
# wall short. Phase breakdown across PER_BEAM (the empirical winner),
# SHARED_2L_1POOL, SHARED_3L_1POOL, SHARED_2L_DEC_TAIL, picker.
uv run python benchmarks/bs_kernel/profile_phases.py \
    --K 4 \
    --L_p 8192 \
    --B 4 \
    --max_new 128 \
    --modes PER_BEAM SHARED_2L_1POOL SHARED_3L_1POOL SHARED_2L_DEC_TAIL picker

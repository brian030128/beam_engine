#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bsk_tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/run_bsk_tests_%j.out
#SBATCH --error=slurm/logs/run_bsk_tests_%j.err

# Run the bs_kernel test suite (standalone scripts, not pytest) after the
# XPROMPT_DEC_TAIL picker change.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf

for t in tests/test_bs_kernel.py tests/test_bs_kernel_cache.py tests/test_baselines.py; do
  echo "================ $t ================"
  uv run python "$t" && echo "PASS: $t" || echo "FAIL($?): $t"
done
echo "All test scripts done."

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=v3_build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/v3_build_%j.out
#SBATCH --error=slurm/logs/v3_build_%j.err

set -euo pipefail

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-unset}"
date

cd /home/u4320956/beam_engine-v3

source /etc/profile.d/lmod.sh
module load cuda/12.4

echo
echo "=== nvcc ==="
nvcc --version

echo
echo "=== uv sync (rebuild venv from v3 pyproject) ==="
# Use HPC-built Python 3.11.11 because /usr/bin/python3.11 ships without
# Python.h headers, which triton's JIT needs to compile its CUDA utils.
HPC_PY=/work/HPC_software/LMOD/biology/packages/Python/Python_v3.11.11/bin/python3.11
rm -rf .venv
~/.local/bin/uv sync --python "$HPC_PY"

echo
echo "=== Sanity check ==="
source .venv/bin/activate
python - <<'PY'
import torch, flashinfer
print("torch:        ", torch.__version__)
print("cuda built:   ", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count: ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0 name:", torch.cuda.get_device_name(0))
print("flashinfer:   ", flashinfer.__file__)
PY

echo
echo "=== Done ==="
date

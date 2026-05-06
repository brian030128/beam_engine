#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bs_kernel_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/bs_kernel_bench_%j.out
#SBATCH --error=slurm/logs/bs_kernel_bench_%j.err

set -euo pipefail

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-unset}"
echo "GRID_MODE: ${GRID_MODE:-quick}"
date

cd /home/u4320956/beam_engine-v3

source /etc/profile.d/lmod.sh
module load cuda/12.4

# Note: NOT loading cuda-compat shim here. The kernel driver is 550.x
# (CUDA 12.4) and SM90/Hopper kernel features need a 570+ kernel driver,
# which user-mode forward-compat cannot supply. We let cuDriverGetVersion
# report the real 12.4 so flashinfer's auto-backend (with our patched
# is_sm90a_supported in 3rdparty/flashinfer/flashinfer/utils.py) falls
# back to FA2 — the only path that runs on this driver.
source .venv/bin/activate

echo
echo "=== Driver version probe ==="
python -c "
import ctypes
libcuda = ctypes.CDLL('libcuda.so.1')
v = ctypes.c_int(0)
libcuda.cuDriverGetVersion(ctypes.byref(v))
print('cuDriverGetVersion =', v.value, '(expected 12040 for fa2 fallback)')
"

echo
echo "=== nvidia-smi (informational) ==="
nvidia-smi

echo
echo "=== Running run_all.sh ==="
# SLURM has already pinned CUDA_VISIBLE_DEVICES to one GPU.
# Passing "0" selects "the GPU SLURM gave us" within that visible set.
./benchmarks/bs_kernel/run_all.sh 0 "${GRID_MODE:-quick}"

echo
echo "=== Done ==="
date

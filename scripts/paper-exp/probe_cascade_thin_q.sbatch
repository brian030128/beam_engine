#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=probe_thin_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/probe_thin_q_%j.out
#SBATCH --error=slurm/logs/probe_thin_q_%j.err

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

OUT_DIR="benchmarks/bs_kernel/results/probe_thin_q"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== probe cascade thin-Q L1 marginal cost $(date -Iseconds) ==="
    echo
} | tee "$LOG"

uv run python benchmarks/bs_kernel/probe_cascade_thin_q.py 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"

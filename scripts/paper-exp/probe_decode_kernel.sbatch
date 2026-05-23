#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=probe_dec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/probe_dec_%j.out
#SBATCH --error=slurm/logs/probe_dec_%j.err

# Direct FlashInfer decode-kernel probe — measures wall time at varying
# (R, L_kv) to characterise per-CTA serial setup (issue 1) and
# effective HBM bandwidth at small R (issue 2).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT="benchmarks/bs_kernel/results/paper-exp/probe_decode_kernel.csv"
mkdir -p "$(dirname "$OUT")" slurm/logs

uv run python scripts/paper-exp/probe_decode_kernel.py --out "$OUT"

echo
echo "=== analysis ==="
echo "Effective BW (GB/s) should rise with R (more CTAs → more parallel HBM"
echo "loads) until it saturates near peak ~1.3 TB/s. Wall time vs R at"
echo "fixed L_kv has slope = (κ_per_cta + L_kv·ρ_d_inner) per wave;"
echo "intercept at R→0 is ν_decode_launch."

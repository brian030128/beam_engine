#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=valid_dpx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:08:00
#SBATCH --output=slurm/logs/valid_dpx_%j.out
#SBATCH --error=slurm/logs/valid_dpx_%j.err

# Validate that bumping dual_pool_extra_us in the H100 cache flips the
# auto-picker from 3L2P to 3L1P at the K=64 / L_p=8192 / B=32 cell.
# Runs only mode=auto (compare to the prior 'auto' run from
# diagnose_cell-20260508-004515 which picked 3L2P 239/255).

set -euo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/validate_dpx-${TS}"
mkdir -p "$RESULTS_DIR"

echo "=== cache contents ==="
cat ~/.cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json
echo

uv run python benchmarks/bs_kernel/diagnose_cell.py run \
    --mode auto \
    --out "$RESULTS_DIR/auto.json"

echo
echo "=== picks histogram ==="
python3 - <<'PY'
import json, statistics
from collections import Counter
from pathlib import Path
import glob
d = sorted(glob.glob("benchmarks/bs_kernel/results/validate_dpx-*"))[-1]
auto = json.loads((Path(d) / "auto.json").read_text())
picks = auto["picks"]
print(f"Result dir: {d}")
print(f"steps: {len(picks)}")
hist = Counter(p["strategy"] for p in picks)
for s, n in hist.most_common():
    print(f"  {s:22s}  {n:4d}")
decode = auto["decode_step_ms"]
print(f"decode median: {statistics.median(decode):.2f}ms")
print(f"decode total:  {sum(decode):.1f}ms")
PY

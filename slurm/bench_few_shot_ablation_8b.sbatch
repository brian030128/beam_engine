#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=fs_abl_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/bench_few_shot_ablation_8b_%j.out
#SBATCH --error=slurm/logs/bench_few_shot_ablation_8b_%j.err

# Forced-strategy ablation for bs_kernel on multi_few_shot paper-exact
# (8B). We want to know whether bs_kernel's loss to fasttree on this
# cell is a picker problem (some forced strategy would beat fasttree)
# or a kernel problem (flashinfer cascade is intrinsically slower than
# fasttree's Triton kernel for this 8-group × 16-query × ~5.9K-shared
# shape on 8B).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.1-8B

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/few_shot_ablation_8b-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== few_shot ablation 8B started $(date -Iseconds) ==="
    echo "scenario: multi_few_shot (paper-exact, B=8 K=16 L_shared~5861)"
    echo "methods:  fasttree + bs_kernel(picker) + 6 forced bsk variants"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_few_shot \
    --methods fasttree bs_kernel \
              bsk_per_beam bsk_2l_1p bsk_3l_1p \
              bsk_2l_dt bsk_3l_dt bsk_2l_2p \
    --paper-exact \
    --max-new 256 --repeat 1 \
    --out "$RESULTS_DIR/ablation.csv" \
    2>&1 | tee -a "$LOG"

echo
echo "=== picker vs forced comparison ==="
uv run python - <<PY 2>&1 | tee "$RESULTS_DIR/picker_vs_forced.txt"
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("$RESULTS_DIR/ablation.csv").open()))
labels = {
    "fasttree":     "fasttree (ref)",
    "bs_kernel":    "bs_kernel picker",
    "bsk_per_beam": "  forced PER_BEAM",
    "bsk_2l_1p":    "  forced SHARED_2L_1POOL",
    "bsk_3l_1p":    "  forced SHARED_3L_1POOL",
    "bsk_2l_dt":    "  forced SHARED_2L_DEC_TAIL",
    "bsk_3l_dt":    "  forced SHARED_3L_DEC_TAIL",
    "bsk_2l_2p":    "  forced SHARED_2L_2POOL",
}
order = ("fasttree","bs_kernel","bsk_per_beam","bsk_2l_1p","bsk_3l_1p",
         "bsk_2l_dt","bsk_3l_dt","bsk_2l_2p")
ft = pk = None
for r in rows:
    if r["method"] == "fasttree": ft = float(r["decode_total_ms"])
    if r["method"] == "bs_kernel": pk = float(r["decode_total_ms"])
print(f"\n========== multi_few_shot paper-exact (8B) ==========")
print(f"{'method':<32} {'decode_total':>12} {'plan_total':>11} {'fwd_total':>11} "
      f"{'p50/step':>9} {'vs ft':>8} {'vs picker':>10}")
for m in order:
    cell = next((r for r in rows if r["method"] == m), None)
    if not cell:
        print(f"{labels[m]:<32} (no row)"); continue
    dt = float(cell["decode_total_ms"]); pt = float(cell["plan_total_ms"])
    fwt = float(cell["forward_total_ms"]); p50 = float(cell["decode_p50_ms"])
    vs_ft = f"{(dt-ft)/ft*100:+5.1f}%" if ft else "—"
    vs_pk = f"{(dt-pk)/pk*100:+5.1f}%" if pk and m != "bs_kernel" else "—"
    print(f"{labels[m]:<32} {dt:>12.1f} {pt:>11.1f} "
          f"{fwt:>11.1f} {p50:>9.2f} {vs_ft:>8} {vs_pk:>10}")
# Forward-only delta vs fasttree:
print()
print(f"{'method':<32} {'fwd_total':>11} {'fwd vs ft':>11} {'fwd/step':>9}")
for m in order:
    cell = next((r for r in rows if r["method"] == m), None)
    if not cell: continue
    fwt = float(cell["forward_total_ms"])
    fmean = float(cell["forward_mean_ms"])
    ft_fwt = next((float(r["forward_total_ms"]) for r in rows if r["method"]=="fasttree"), None)
    delta = f"{(fwt-ft_fwt)/ft_fwt*100:+5.1f}%" if ft_fwt else "—"
    print(f"{labels[m]:<32} {fwt:>11.1f} {delta:>11} {fmean:>9.2f}")
PY

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

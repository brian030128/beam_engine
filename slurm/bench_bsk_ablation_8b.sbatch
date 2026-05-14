#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=bsk_abl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_bsk_ablation_8b_%j.out
#SBATCH --error=slurm/logs/bench_bsk_ablation_8b_%j.err

# Picker-vs-kernel ablation for bs_kernel on the 3 cells where bs_kernel
# trailed fasttree on Llama-3.1-8B at K=16 (phase 1):
#   multi_few_shot, multi_chain_reasoning, multi_document
#
# Methods compared (all on the same workload):
#   fasttree                — reference baseline
#   bs_kernel               — production picker (cost-model chooses)
#   bsk_per_beam            — forced PER_BEAM
#   bsk_2l_1p               — forced SHARED_2L_1POOL
#   bsk_3l_1p               — forced SHARED_3L_1POOL
#   bsk_2l_dt               — forced SHARED_2L_DEC_TAIL
#   bsk_3l_dt               — forced SHARED_3L_DEC_TAIL
#   bsk_2l_2p               — forced SHARED_2L_2POOL (non-default-set)
#
# Reading: if any forced strategy < picker, the picker is suboptimal for
# the cell. If picker matches the best forced strategy and STILL loses
# to fasttree, the gap is in our flashinfer-cascade kernel for that
# shape (not picker error).

set -uo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.1-8B

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/bsk_ablation_8b-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== bsk ablation 8B K=16 started $(date -Iseconds) ==="
    echo "scenarios: multi_few_shot multi_chain_reasoning multi_document"
    echo "methods:   fasttree + bs_kernel(picker) + 6 forced bsk variants"
    echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_few_shot multi_chain_reasoning multi_document \
    --methods fasttree bs_kernel \
              bsk_per_beam bsk_2l_1p bsk_3l_1p \
              bsk_2l_dt bsk_3l_dt bsk_2l_2p \
    --k-override 16 \
    --max-new 256 \
    --repeat 1 \
    --out "$RESULTS_DIR/ablation.csv" \
    2>&1 | tee -a "$LOG"

echo
echo "=== picker vs forced comparison ==="
PAPER="$RESULTS_DIR/picker_vs_forced.txt"
uv run python - <<PY 2>&1 | tee "$PAPER"
import csv
from pathlib import Path

csv_path = Path("$RESULTS_DIR/ablation.csv")
rows = list(csv.DictReader(csv_path.open()))

# Group by scenario; within each, sort methods by decode_total.
scenarios = ("multi_few_shot", "multi_chain_reasoning", "multi_document")
method_order = ("fasttree", "bs_kernel",
                "bsk_per_beam", "bsk_2l_1p", "bsk_3l_1p",
                "bsk_2l_dt", "bsk_3l_dt", "bsk_2l_2p")
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

for scen in scenarios:
    cells = [r for r in rows if r["scenario"] == scen]
    if not cells:
        print(f"\n[{scen}] no rows"); continue
    fasttree_row = next((r for r in cells if r["method"] == "fasttree"), None)
    picker_row   = next((r for r in cells if r["method"] == "bs_kernel"), None)
    ft = float(fasttree_row["decode_total_ms"]) if fasttree_row else None
    pk = float(picker_row["decode_total_ms"]) if picker_row else None

    print(f"\n========== {scen} (Llama-3.1-8B, K=16) ==========")
    print(f"{'method':<32} {'decode_total':>12} {'plan_total':>11} "
          f"{'fwd_total':>11} {'p50/step':>9} {'vs ft':>8} {'vs picker':>10}")
    for m in method_order:
        cell = next((r for r in cells if r["method"] == m), None)
        if not cell:
            print(f"{labels[m]:<32} {'(no row)':>12}")
            continue
        dt = float(cell["decode_total_ms"])
        pt = float(cell["plan_total_ms"])
        fwt = float(cell["forward_total_ms"])
        p50 = float(cell["decode_p50_ms"])
        vs_ft = f"{(dt-ft)/ft*100:+5.1f}%" if ft else "—"
        vs_pk = f"{(dt-pk)/pk*100:+5.1f}%" if pk and m != "bs_kernel" else "—"
        print(f"{labels[m]:<32} {dt:>12.1f} {pt:>11.1f} "
              f"{fwt:>11.1f} {p50:>9.2f} {vs_ft:>8} {vs_pk:>10}")

    # Verdict.
    forced_methods = [m for m in method_order
                      if m.startswith("bsk_") and any(r["method"] == m for r in cells)]
    if forced_methods:
        forced_rows = [next(r for r in cells if r["method"] == m) for m in forced_methods]
        best_forced = min(forced_rows, key=lambda r: float(r["decode_total_ms"]))
        best_forced_total = float(best_forced["decode_total_ms"])
        best_forced_name  = best_forced["method"]
        if pk is not None:
            if best_forced_total < pk - 5.0:
                verdict = (f"PICKER SUBOPTIMAL: {best_forced_name} "
                           f"({best_forced_total:.0f}) beats picker ({pk:.0f}) "
                           f"by {pk - best_forced_total:.0f} ms")
            elif ft is not None and best_forced_total > ft + 5.0:
                verdict = (f"KERNEL LIMITATION: no forced strategy beats "
                           f"fasttree ({ft:.0f}). Best bsk = "
                           f"{best_forced_name} ({best_forced_total:.0f}).")
            else:
                verdict = (f"PICKER OK: picker matches best forced "
                           f"({best_forced_name})")
            print(f"  >>> {verdict}")

print(f"\nfull CSV: {csv_path}")
PY

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

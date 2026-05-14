#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=phase_bsk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/bench_phase_paper_bsk_only_%j.out
#SBATCH --error=slurm/logs/bench_phase_paper_bsk_only_%j.err

# bs_kernel-only re-run of the phase_paper config, after autotune on 8B
# rewrote dec_tail_extra_us (5 → 20) and share_extra_us (100 → 0). The
# fasttree/paged/mlca rows from phase_paper-20260514-014245 are
# unchanged by the autotune, so we only re-run bs_kernel and merge
# the new rows back into the original CSVs.

set -uo pipefail

cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/phase_paper_bsk-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

ORIG_DIR="benchmarks/bs_kernel/results/phase_paper-20260514-014245"

{
    echo "=== phase_paper bsk-only re-run $(date -Iseconds) ==="
    echo "Target merge dir: $ORIG_DIR"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    cat ~/.cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json
    echo
} | tee "$LOG"

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    export BE_MODEL="meta-llama/${model_tag}"
    csv="$RESULTS_DIR/${model_tag}.csv"
    echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_level_system multi_few_shot \
                    multi_chain_reasoning multi_document \
        --methods bs_kernel \
        --k-override 16,32 \
        --max-new 256 \
        --repeat 1 \
        --out "$csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

# Merge — drop bs_kernel rows from each ORIG CSV, append new bs_kernel rows.
echo "=== merging bs_kernel rows into $ORIG_DIR ==="
for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    orig="$ORIG_DIR/${model_tag}.csv"
    new="$RESULTS_DIR/${model_tag}.csv"
    merged="$ORIG_DIR/${model_tag}.csv.merged"
    backup="$ORIG_DIR/${model_tag}.csv.bak.before_bsk_remerge"

    # Backup
    cp "$orig" "$backup"
    # Keep all non-bs_kernel rows from orig
    awk -F, 'NR==1 || $3!="bs_kernel"' "$orig" > "$merged"
    # Append new bs_kernel rows (data rows only)
    awk -F, 'NR>1 && $3=="bs_kernel"' "$new" >> "$merged"
    mv "$merged" "$orig"
    echo "  $model_tag merged ($(wc -l < $orig) rows total)"
done

# Regenerate paper table.
PAPER="$ORIG_DIR/paper_table.txt"
echo "=== regenerated paper_table.txt ==="
uv run python - <<PY 2>&1 | tee "$PAPER"
import csv, sys
from pathlib import Path
dir_ = Path("$ORIG_DIR")
files = sorted(dir_.glob("Llama-*.csv"))
rows = []
for f in files:
    model = f.stem
    with f.open() as fh:
        for r in csv.DictReader(fh):
            r["model"] = model
            rows.append(r)
def fmt(x): return f"{float(x):7.1f}"
for k_override in ("16", "32"):
    phase = "phase 1" if k_override == "16" else "phase 0"
    print(f"\n========== K={k_override}  ({phase}) ==========")
    print(f"{'model':<14} {'scenario':<24} {'method':<10} "
          f"{'decode_total':>12} {'plan_total':>11} {'fwd_total':>11} "
          f"{'plan%':>6} {'fwd%':>6} {'p50/step':>9}")
    for model in sorted({r["model"] for r in rows}):
        for scen in ("multi_level_system","multi_few_shot",
                     "multi_chain_reasoning","multi_document"):
            for meth in ("fasttree","bs_kernel","paged","mlca"):
                cell = [r for r in rows
                        if r["model"]==model
                        and r["k_override"]==k_override
                        and r["scenario"]==scen
                        and r["method"]==meth]
                if not cell: continue
                r = cell[0]
                dt = float(r["decode_total_ms"]); pt = float(r["plan_total_ms"])
                ft = float(r["forward_total_ms"]); p50 = float(r["decode_p50_ms"])
                pct_p = 100.0 * pt / dt if dt > 0 else 0.0
                pct_f = 100.0 * ft / dt if dt > 0 else 0.0
                print(f"{model:<14} {scen:<24} {meth:<10} "
                      f"{fmt(dt):>12} {fmt(pt):>11} {fmt(ft):>11} "
                      f"{pct_p:>5.1f}% {pct_f:>5.1f}% {p50:>9.2f}")
print(f"\nfull CSVs: $ORIG_DIR")
PY

echo "Done. Backups at: $ORIG_DIR/*.bak.before_bsk_remerge" | tee -a "$LOG"

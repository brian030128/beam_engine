#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=fs_B_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/bench_few_shot_B_sweep_%j.out
#SBATCH --error=slurm/logs/bench_few_shot_B_sweep_%j.err

# multi_few_shot paper-exact at larger B, to test the hypothesis that
# bs_kernel's loss to fasttree at B=8 K=16 is a small-batch /
# borderline-CTA-count corner of flashinfer's fused cascade.
# At B=16 and B=32 the cascade prefill stage should have enough CTAs
# to fully saturate the GPU, eliminating the per-call setup overhead
# that gives fasttree the edge.
#
# All runs use --warmup --repeat 3 → median of 3 warm iters.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/few_shot_B_sweep-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== few_shot_B_sweep started $(date -Iseconds) ==="
    echo "scenario:    multi_few_shot (paper-exact, K=16)"
    echo "B values:    8 (paper default), 16, 32"
    echo "methods:     fasttree bs_kernel paged"
    echo "models:      Llama-3.2-1B, Llama-3.1-8B"
    echo "--warmup --repeat 3 --max-new 256"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    export BE_MODEL="meta-llama/${model_tag}"
    for B in 8 16 32; do
        csv="$RESULTS_DIR/${model_tag}_B${B}.csv"
        echo "############ ${model_tag}  B=${B} ############" | tee -a "$LOG"
        uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
            --scenarios multi_few_shot \
            --methods fasttree bs_kernel paged \
            --paper-exact --b-override $B \
            --max-new 256 --repeat 3 --warmup \
            --out "$csv" \
            2>&1 | tee -a "$LOG"
        echo | tee -a "$LOG"
    done
done

PAPER="$RESULTS_DIR/paper_table.txt"
uv run python - <<PY 2>&1 | tee "$PAPER"
import csv, statistics, re
from pathlib import Path
dir_ = Path("$RESULTS_DIR")
rows = []
for f in sorted(dir_.glob("Llama-*_B*.csv")):
    m = re.match(r"(Llama[^_]+)_B(\d+)", f.stem)
    if not m: continue
    model, B = m.group(1), int(m.group(2))
    with f.open() as fh:
        for r in csv.DictReader(fh):
            r["model"] = model; r["B"] = B
            rows.append(r)
def med(rs, k): return statistics.median([float(r[k]) for r in rs]) if rs else 0.0
print("\n========== multi_few_shot paper-exact B sweep (warmup, median of 3) ==========")
print(f"{'model':<14} {'B':<3} {'method':<10} "
      f"{'decode_total':>12} {'plan_total':>11} {'fwd_total':>11} "
      f"{'plan%':>6} {'fwd%':>6} {'p50/step':>9}")
for model in sorted({r["model"] for r in rows}):
    for B in (8, 16, 32):
        for meth in ("fasttree","bs_kernel","paged"):
            cells = [r for r in rows if r["model"]==model and r["B"]==B and r["method"]==meth]
            if not cells: continue
            dt = med(cells, "decode_total_ms"); pt = med(cells, "plan_total_ms")
            ft = med(cells, "forward_total_ms"); p50 = med(cells, "decode_p50_ms")
            pct_p = 100.0*pt/dt if dt>0 else 0.0
            pct_f = 100.0*ft/dt if dt>0 else 0.0
            print(f"{model:<14} {B:<3} {meth:<10} "
                  f"{dt:>12.1f} {pt:>11.1f} {ft:>11.1f} "
                  f"{pct_p:>5.1f}% {pct_f:>5.1f}% {p50:>9.2f}")
print(f"\nfull CSVs: $RESULTS_DIR")
PY

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

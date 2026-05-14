#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=paper_tbl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm/logs/bench_paper_table_h100_%j.out
#SBATCH --error=slurm/logs/bench_paper_table_h100_%j.err

# Single-script reproduction of the H100 paper table (paper_exact_warm +
# few_shot_B_sweep), with mlca added to the method set.
#
# Cells (max_new=256, --warmup --repeat 3, median of 3 warm iters):
#
#   scenario              | model      | (B, K)
#   ----------------------|------------|--------
#   multi_level_system    | 1B / 8B    | (4, 32)   [paper-exact natural]
#   multi_chain_reasoning | 1B / 8B    | (32, 16)  [K=16 override]
#   multi_few_shot        | 1B         | (32, 16)  [b_override=32]
#   multi_few_shot        | 8B         | (16, 16)  [b_override=16, OOM at 32]
#
# Methods: fasttree bs_kernel paged mlca
# Phase timings (plan_total / forward_total) come from bench_sglang_e2e.py
# `--return_phase_timings` path and are aggregated below.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="${RESULTS_DIR_OVERRIDE:-benchmarks/bs_kernel/results/paper_table_h100-${TS}}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"
# Set RESUME_ONLY_CHAIN=1 to re-use any existing level_system/few_shot CSVs
# in $RESULTS_DIR and only run the chain_reasoning passes.
RESUME_ONLY_CHAIN="${RESUME_ONLY_CHAIN:-0}"

METHODS="fasttree bs_kernel paged mlca"

{
    echo "=== paper_table_h100 started $(date -Iseconds) ==="
    echo "methods:  ${METHODS}"
    echo "models:   Llama-3.2-1B, Llama-3.1-8B"
    echo "max_new=256, --warmup --repeat 3 (median of 3)"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

run_cell () {
    # run_cell <model_tag> <csv_out> <extra-args...>
    local model_tag="$1"; shift
    local csv="$1"; shift
    export BE_MODEL="meta-llama/${model_tag}"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS \
        --paper-exact \
        --max-new 256 --repeat 3 --warmup \
        --out "$csv" \
        "$@" \
        2>&1 | tee -a "$LOG"
}

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    echo "############ MODEL: ${model_tag} ############" | tee -a "$LOG"

    # Pass 1: multi_level_system (natural B=4, K=32)
    csv_a="$RESULTS_DIR/${model_tag}_level_system.csv"
    if [[ "$RESUME_ONLY_CHAIN" == "1" && -f "$csv_a" ]]; then
        echo "  pass 1: SKIP (using existing $csv_a)" | tee -a "$LOG"
    else
        echo "  pass 1: multi_level_system  (B=4, K=32 natural)" | tee -a "$LOG"
        run_cell "$model_tag" "$csv_a" \
            --scenarios multi_level_system
    fi

    # Pass 2a: multi_chain_reasoning at K=16 — non-mlca methods.
    # mlca crashes the CUDA context on chain_reasoning's two-stage spec, so
    # we run it in its own subprocess (pass 2b) to keep the others' CSV intact.
    csv_b="$RESULTS_DIR/${model_tag}_chain_reasoning.csv"
    echo "  pass 2a: multi_chain_reasoning  (B=32, K=16)  [fasttree bs_kernel paged]" | tee -a "$LOG"
    METHODS_NON_MLCA="fasttree bs_kernel paged"
    export BE_MODEL="meta-llama/${model_tag}"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS_NON_MLCA \
        --paper-exact \
        --max-new 256 --repeat 3 --warmup \
        --scenarios multi_chain_reasoning \
        --k-override 16 \
        --out "$csv_b" \
        2>&1 | tee -a "$LOG"

    csv_b2="$RESULTS_DIR/${model_tag}_chain_reasoning_mlca.csv"
    echo "  pass 2b: multi_chain_reasoning  (B=32, K=16)  [mlca, may CUDA-fail]" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods mlca \
        --paper-exact \
        --max-new 256 --repeat 3 --warmup \
        --scenarios multi_chain_reasoning \
        --k-override 16 \
        --out "$csv_b2" \
        2>&1 | tee -a "$LOG" || echo "  (mlca chain_reasoning failed; continuing)" | tee -a "$LOG"

    # Pass 3: multi_few_shot at (B=32, K=16) for 1B, (B=16, K=16) for 8B
    if [[ "$model_tag" == "Llama-3.2-1B" ]]; then
        few_shot_B=32
    else
        few_shot_B=16
    fi
    csv_c="$RESULTS_DIR/${model_tag}_few_shot.csv"
    if [[ "$RESUME_ONLY_CHAIN" == "1" && -f "$csv_c" ]]; then
        echo "  pass 3: SKIP (using existing $csv_c)" | tee -a "$LOG"
    else
        echo "  pass 3: multi_few_shot  (B=${few_shot_B}, K=16)" | tee -a "$LOG"
        run_cell "$model_tag" "$csv_c" \
            --scenarios multi_few_shot \
            --b-override "$few_shot_B"
    fi

    # Merge into a single ${model_tag}.csv (header from first, drop from rest).
    # Skip any intermediate CSV that didn't get written (e.g. mlca crash).
    merged="$RESULTS_DIR/${model_tag}.csv"
    cp "$csv_a" "$merged"
    [[ -f "$csv_b"  ]] && tail -n +2 "$csv_b"  >> "$merged"
    [[ -f "$csv_b2" ]] && tail -n +2 "$csv_b2" >> "$merged"
    [[ -f "$csv_c"  ]] && tail -n +2 "$csv_c"  >> "$merged"
    echo "  merged → $merged" | tee -a "$LOG"
    echo | tee -a "$LOG"
done

# Aggregate paper table (median over 3 warm iters per cell).
PAPER="$RESULTS_DIR/paper_table.txt"
RESULTS_DIR_ABS="$(realpath "$RESULTS_DIR")"
uv run python - "$RESULTS_DIR_ABS" <<'PY' 2>&1 | tee "$PAPER"
import csv, statistics, sys
from pathlib import Path

dir_ = Path(sys.argv[1])
all_rows = []
for f in sorted(dir_.glob("Llama-*.csv")):
    # Skip per-pass intermediates; only consume the merged per-model CSV.
    if any(tag in f.name for tag in ("_level_system.csv",
                                     "_chain_reasoning.csv",
                                     "_few_shot.csv")):
        continue
    model = f.stem
    with f.open() as fh:
        for r in csv.DictReader(fh):
            r["model"] = model
            all_rows.append(r)

def med(rows, key):
    vals = [float(r[key]) for r in rows]
    return statistics.median(vals) if vals else 0.0

# Display tag including (B, K) and the methods row order matching the
# original paper table.
SCENARIOS = ("multi_level_system", "multi_few_shot", "multi_chain_reasoning")
METHODS   = ("bs_kernel", "paged", "fasttree", "mlca")

print("\n========== H100 paper table (warmup, median of 3) ==========")
print(f"{'model':<14} {'scenario':<28} {'method':<10} "
      f"{'decode_total':>12} {'plan_total':>11} {'fwd_total':>11} "
      f"{'plan%':>6} {'fwd%':>6} {'p50/step':>9}")

for model in sorted({r["model"] for r in all_rows}):
    for scen in SCENARIOS:
        cells_scen = [r for r in all_rows
                      if r["model"] == model and r["scenario"] == scen]
        if not cells_scen:
            continue
        # Derive K used from k_override (0 ⇒ scenario's natural K).
        NATURAL_K = {"multi_level_system": 32,
                     "multi_few_shot": 16,
                     "multi_chain_reasoning": 4}
        k_used = int(cells_scen[0].get("k_override", 0)) or NATURAL_K[scen]
        B_used = int(cells_scen[0].get("n_leaves", 0)) // k_used
        tag = f"{scen} (B={B_used}, K={k_used})"

        for meth in METHODS:
            cells = [r for r in cells_scen if r["method"] == meth]
            if not cells:
                continue
            dt = med(cells, "decode_total_ms")
            pt = med(cells, "plan_total_ms")
            ft = med(cells, "forward_total_ms")
            p50 = med(cells, "decode_p50_ms")
            pct_p = 100.0 * pt / dt if dt > 0 else 0.0
            pct_f = 100.0 * ft / dt if dt > 0 else 0.0
            print(f"{model:<14} {tag:<28} {meth:<10} "
                  f"{dt:>12.1f} {pt:>11.1f} {ft:>11.1f} "
                  f"{pct_p:>5.1f}% {pct_f:>5.1f}% {p50:>9.2f}")

# Speedup summary table — bs_kernel vs fasttree / paged / mlca.
print("\n========== bs_kernel speedups ==========")
print(f"{'scenario':<28} {'model':<14} "
      f"{'vs fasttree':>12} {'vs paged':>10} {'vs mlca':>10}")
for model in sorted({r["model"] for r in all_rows}):
    for scen in SCENARIOS:
        cells_scen = [r for r in all_rows
                      if r["model"] == model and r["scenario"] == scen]
        if not cells_scen:
            continue
        def med_for(meth, key="decode_total_ms"):
            cells = [r for r in cells_scen if r["method"] == meth]
            return med(cells, key) if cells else 0.0
        bsk = med_for("bs_kernel")
        if bsk <= 0:
            continue
        def sp(meth):
            x = med_for(meth)
            return f"{x / bsk:.2f}x" if x > 0 else "   -- "
        print(f"{scen:<28} {model:<14} "
              f"{sp('fasttree'):>12} {sp('paged'):>10} {sp('mlca'):>10}")

print(f"\nfull CSVs: {dir_}")
PY

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

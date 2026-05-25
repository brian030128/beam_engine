#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp3l_mfs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/bench_xprompt_3l_mfs_%j.out
#SBATCH --error=slurm/logs/bench_xprompt_3l_mfs_%j.err

# Cross-prompt 3L: L0 = ONE group of all B*K beams reading the shared sys
# ONCE, L1 = B per-prompt bundle groups, L2 = per-beam tails. This is the
# 3L structure with a real read saving (sys read 1× instead of B×).
# Compare vs the same-TreeSpec good-2L baseline, on BOTH timing and
# token-identity (the cross-prompt cascade must be numerically equivalent).
#   cell: multi_few_shot K=32 B=4 (128 leaves), max_new=256, --tree-root
#   Run X: NO knob          → bsk_2l_dt / bsk_2l_1p  (good-2L baseline)
#   Run Y: CROSS_PROMPT_3L=1 → bs_kernel             (cross-prompt 3L)

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL=${BE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
export BS_KERNEL_TRACE_PLAN=1

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/xprompt_3l_mfs-${TS}"
BEAMS_DIR="$RESULTS_DIR/beams"
mkdir -p "$BEAMS_DIR"
LOG="$RESULTS_DIR/bench.log"
export BE_DUMP_BEAMS_DIR="$BEAMS_DIR"

{
    echo "=== cross-prompt 3L multi_few_shot $(date -Iseconds) ==="
    echo "model: $BE_MODEL ; cell: multi_few_shot K=32 B=4 max_new=256 (--tree-root)"
} | tee "$LOG"

echo "=== Run X: --tree-root, NO knob → good-2L baseline ===" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_few_shot --tree-root \
    --methods bsk_2l_dt bsk_2l_1p \
    --k-override 32 --b-override 4 \
    --max-new 256 --repeat 3 --warmup \
    --out "$RESULTS_DIR/run_x_2l.csv" \
    2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== Run Y: --tree-root + CROSS_PROMPT_3L=1 → cross-prompt 3L ===" | tee -a "$LOG"
BS_KERNEL_CROSS_PROMPT_3L=1 \
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_few_shot --tree-root \
    --methods bs_kernel \
    --k-override 32 --b-override 4 \
    --max-new 256 --repeat 3 --warmup \
    --out "$RESULTS_DIR/run_y_xp3l.csv" \
    2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== token-identity check: cross-prompt-3L (bs_kernel) vs good-2L (bsk_2l_dt) ===" | tee -a "$LOG"
uv run python - "$BEAMS_DIR" <<'PY' 2>&1 | tee -a "$LOG"
import json, os, sys
d = sys.argv[1]
ref = json.load(open(os.path.join(d, "multi_few_shot__bsk_2l_dt.json")))["leaf_token_ids"]
xp  = json.load(open(os.path.join(d, "multi_few_shot__bs_kernel.json")))["leaf_token_ids"]
same = ref == xp
nmis = sum(1 for a,b in zip(ref,xp) if a!=b)
print(f"leaves: ref={len(ref)} xp={len(xp)}  identical={same}  mismatched_leaves={nmis}")
print("RESULT: TOKEN-IDENTICAL" if same else "RESULT: MISMATCH (cross-prompt 3L is INCORRECT)")
PY

echo | tee -a "$LOG"; echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp3l_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/verify_xprompt_3l_dt_%j.out
#SBATCH --error=slurm/logs/verify_xprompt_3l_dt_%j.err

# Correctness: dump beams for 2L baseline, cross-prompt 3L_1p, cross-prompt
# 3L_dt at the 20k/20k B=16 K=16 cell and compare per-leaf token sequences.
# 3L_dt should match 3L_1p within the fp-noise envelope (= the leaf-mismatch
# count between two correct kernels 2L_dt vs 2L_1p, ~17/256 historically).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL=${BE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
export BE_SYS_LEN=20000
export BE_FEWSHOT_LEN=20000

TS=$(date +%Y%m%d-%H%M%S)
RD="benchmarks/bs_kernel/results/xprompt_3l_20k_verify-${TS}"
mkdir -p "$RD/beams"
LOG="$RD/bench.log"
export BE_DUMP_BEAMS_DIR="$RD/beams"
COMMON="--scenarios multi_few_shot --tree-root --k-override 16 --b-override 16 --max-new 128 --repeat 1"
echo "=== verify cross-prompt 3L_dt $(date -Iseconds) ===" | tee "$LOG"

uv run python benchmarks/bs_kernel/bench_sglang_e2e.py $COMMON --methods bsk_2l_dt 2>&1 | tee -a "$LOG"
BS_KERNEL_CROSS_PROMPT_3L=1 \
  uv run python benchmarks/bs_kernel/bench_sglang_e2e.py $COMMON --methods bs_kernel 2>&1 | tee -a "$LOG"
mv "$RD/beams/multi_few_shot__bs_kernel.json" "$RD/beams/multi_few_shot__xp3l_1p.json" 2>/dev/null || true
BS_KERNEL_CROSS_PROMPT_3L=1 BS_KERNEL_XP_DEC_TAIL=1 \
  uv run python benchmarks/bs_kernel/bench_sglang_e2e.py $COMMON --methods bs_kernel 2>&1 | tee -a "$LOG"
mv "$RD/beams/multi_few_shot__bs_kernel.json" "$RD/beams/multi_few_shot__xp3l_dt.json" 2>/dev/null || true

echo "" | tee -a "$LOG"
echo "=== compare ===" | tee -a "$LOG"
python3 - "$RD/beams" <<'PY' 2>&1 | tee -a "$LOG"
import json, os, sys
BD = sys.argv[1]
def load(name):
    p = os.path.join(BD, name)
    if not os.path.exists(p): return None
    d = json.load(open(p))
    # stored structure: {"beams": [[token ids per leaf], ...]} or similar
    return d
ref2l = load("multi_few_shot__bsk_2l_dt.json")
p1 = load("multi_few_shot__xp3l_1p.json")
dt = load("multi_few_shot__xp3l_dt.json")
def leaves(d):
    if d is None: return None
    return [tuple(x) for x in d["leaf_token_ids"]]
def cmp(a, b, na, nb):
    if a is None or b is None:
        print(f"  {na} vs {nb}: MISSING ({a is None=}, {b is None=})"); return
    n = min(len(a), len(b))
    mism = sum(1 for i in range(n) if a[i] != b[i])
    print(f"  {na} vs {nb}: leaves={len(a)}/{len(b)}  mismatched={mism}/{n}")
la, lp, ld = leaves(ref2l), leaves(p1), leaves(dt)
cmp(lp, ld, "xp3l_1p", "xp3l_dt")
cmp(la, ld, "2l_dt",  "xp3l_dt")
cmp(la, lp, "2l_dt",  "xp3l_1p")
PY
echo "Done. $RD" | tee -a "$LOG"

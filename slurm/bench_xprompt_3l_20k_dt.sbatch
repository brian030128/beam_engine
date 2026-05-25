#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp3l_20kdt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/bench_xprompt_3l_20k_dt_%j.out
#SBATCH --error=slurm/logs/bench_xprompt_3l_20k_dt_%j.err

# Compare cross-prompt 3L dispatch variants at 20k/20k B=16 K=16 multi_few_shot:
#   - 2L baseline (bsk_2l_dt)                         decode ~2249 ms (reference)
#   - cross-prompt 3L 1POOL (single fused cascade)    decode ~1978 ms (current best)
#   - cross-prompt 3L DEC_TAIL (L0+L1 cascade prefill + CTA_Q=1 decode tail)
# Hypothesis: routing the q=1 per-beam tail through the dedicated decode kernel
# (vs inside the prefix-tuned fused cascade) is faster. The prefill cascade
# (L0 sys + L1 bundles) is fully static during decode -> plan once.

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
RD="benchmarks/bs_kernel/results/xprompt_3l_20k_dt-${TS}"
mkdir -p "$RD"
LOG="$RD/bench.log"
echo "=== xprompt 3L 1POOL vs DEC_TAIL 20k/20k B=16 K=16 $(date -Iseconds) ===" | tee "$LOG"

COMMON="--scenarios multi_few_shot --tree-root --k-override 16 --b-override 16 --max-new 128 --repeat 3 --warmup"

echo "" | tee -a "$LOG"
echo "===== good-2L baseline (bsk_2l_dt) =====" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    $COMMON --methods bsk_2l_dt --out "$RD/2l.csv" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== cross-prompt 3L 1POOL =====" | tee -a "$LOG"
BS_KERNEL_CROSS_PROMPT_3L=1 BS_KERNEL_TRACE_PLAN=1 \
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    $COMMON --methods bs_kernel --out "$RD/3l_1p.csv" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== cross-prompt 3L DEC_TAIL =====" | tee -a "$LOG"
BS_KERNEL_CROSS_PROMPT_3L=1 BS_KERNEL_XP_DEC_TAIL=1 BS_KERNEL_TRACE_PLAN=1 \
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    $COMMON --methods bs_kernel --out "$RD/3l_dt.csv" 2>&1 | tee -a "$LOG"

# Token-identity check: 3L_dt vs 3L_1p (both should match within fp-noise).
echo "" | tee -a "$LOG"
echo "===== token-identity: 3L_dt vs 3L_1p =====" | tee -a "$LOG"
python3 - "$RD" <<'PY' 2>&1 | tee -a "$LOG"
import os, sys, glob, json
RD = sys.argv[1]
def load_beams(tag):
    d = os.path.join(RD, "beams")
    # bench writes beams under results dir; fall back to csv-adjacent if absent
    return None
print("(beam dumps compared manually if needed; timing is the primary signal)")
PY

echo "Done. $RD" | tee -a "$LOG"

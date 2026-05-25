#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp3l_20kopt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/bench_xprompt_3l_20k_opt_%j.out
#SBATCH --error=slurm/logs/bench_xprompt_3l_20k_opt_%j.err

# Timing-only re-run of the 20k/20k B=16 K=16 cell AFTER adding the XPROMPT
# two-tier plan cache (non-leaf pack cache + cascade plan-state cache). The
# pre-opt run had fwd/step 16.46 (2L) vs 14.07 (3L) but plan/step 1.08 vs 4.57
# -> 3L net slower despite the faster forward. Goal: plan/step collapses to
# ~1 ms (>=90% plan-state cache hits) so 3L NETs a win. BS_KERNEL_TRACE_PLAN=1
# reports the cache hit rates.

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
RD="benchmarks/bs_kernel/results/xprompt_3l_20k_opt-${TS}"
mkdir -p "$RD"
LOG="$RD/bench.log"
echo "=== xprompt 3L 20k/20k B=16 K=16 (plan-cache opt) $(date -Iseconds) ===" | tee "$LOG"

COMMON="--scenarios multi_few_shot --tree-root --k-override 16 --b-override 16 --max-new 128 --repeat 3 --warmup"

echo "" | tee -a "$LOG"
echo "===== good-2L baseline (bsk_2l_dt) =====" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    $COMMON --methods bsk_2l_dt --out "$RD/2l.csv" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== cross-prompt 3L (OPT: plan-cached) =====" | tee -a "$LOG"
BS_KERNEL_CROSS_PROMPT_3L=1 BS_KERNEL_TRACE_PLAN=1 \
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    $COMMON --methods bs_kernel --out "$RD/3l_opt.csv" 2>&1 | tee -a "$LOG"

echo "Done. $RD" | tee -a "$LOG"

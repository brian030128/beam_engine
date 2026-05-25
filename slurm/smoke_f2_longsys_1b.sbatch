#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=f2_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/smoke_f2_longsys_1b_%j.out
#SBATCH --error=slurm/logs/smoke_f2_longsys_1b_%j.err

# Smoke the REDEFINED F2 (long shared-sys few_shot) on 1B before any full
# final-results rerun: multi_few_shot --tree-root, NO --paper-exact,
# BE_SYS_LEN=BE_FEWSHOT_LEN=20000, K=128. Confirms (a) it builds at K=128/40k
# prefix without OOM at a fitting B, (b) the bs_kernel picker selects
# XPROMPT_DEC_TAIL, (c) sane timing. Sweeps a couple of B to mirror the ladder.

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
RD="benchmarks/bs_kernel/results/smoke_f2_longsys_1b-${TS}"
mkdir -p "$RD"; LOG="$RD/bench.log"
echo "=== F2 long-sys smoke (1B, sys=fewshot=20k, K=128) $(date -Iseconds) ===" | tee "$LOG"

for B in 8 4 2; do
  echo "===== K=128 B=$B (picker) =====" | tee -a "$LOG"
  BS_KERNEL_TRACE_PLAN=1 \
  uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
      --scenarios multi_few_shot --tree-root --methods bs_kernel paged \
      --k-override 128 --b-override "$B" --max-new 64 --repeat 2 --warmup \
      --out "$RD/B${B}.csv" 2>&1 | tee -a "$LOG"
done
echo "Done. $RD" | tee -a "$LOG"

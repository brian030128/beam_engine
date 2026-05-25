#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pick_20k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/bench_picker_20k_%j.out
#SBATCH --error=slurm/logs/bench_picker_20k_%j.err

# What does the DEFAULT bs_kernel picker choose at the 20k/20k B=16 K=16 cell
# (no CROSS_PROMPT_3L knob)? The picker is per-prompt and has no cross-prompt
# candidate, so it can only pick PER_BEAM / SHARED_2L_* / per-prompt SHARED_3L_*.
# Expectation: SHARED_2L_DEC_TAIL (~2245 ms), leaving the cross-prompt 3L_dt
# win (~1930 ms) on the table. The trace prints the per-step strategy mix.

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
RD="benchmarks/bs_kernel/results/picker_20k-${TS}"
mkdir -p "$RD"
LOG="$RD/bench.log"
echo "=== default picker @ 20k/20k B=16 K=16 $(date -Iseconds) ===" | tee "$LOG"

echo "===== default picker (method=bs_kernel, no knob) =====" | tee -a "$LOG"
BS_KERNEL_TRACE_PLAN=1 \
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_few_shot --tree-root --k-override 16 --b-override 16 \
    --max-new 128 --repeat 3 --warmup --methods bs_kernel \
    --out "$RD/picker.csv" 2>&1 | tee -a "$LOG"

echo "Done. $RD" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp_valid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/validate_xprompt_picker_%j.out
#SBATCH --error=slurm/logs/validate_xprompt_picker_%j.err

# Validate the picker-selected cross-prompt 3L_dt (XPROMPT_DEC_TAIL):
#   (1) default picker (NO env knob) at 20k/20k B=16 K=16 must now pick
#       XPROMPT_DEC_TAIL on steady-state steps → decode ≈ 1930 ms (was 2263).
#   (2) default picker at the SAME cell with BS_KERNEL_NO_XPROMPT=1 must fall
#       back to SHARED_2L_DEC_TAIL (≈ 2245 ms) — confirms the candidate is the
#       only behavioral change and the disable works.
#   (3) unit tests still pass.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL=${BE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}

TS=$(date +%Y%m%d-%H%M%S)
RD="benchmarks/bs_kernel/results/validate_xprompt-${TS}"
mkdir -p "$RD"
LOG="$RD/bench.log"
echo "=== validate XPROMPT picker $(date -Iseconds) ===" | tee "$LOG"

COMMON="--scenarios multi_few_shot --tree-root --k-override 16 --b-override 16 --max-new 128 --repeat 3 --warmup --methods bs_kernel"

echo "" | tee -a "$LOG"
echo "===== (1) default picker @ 20k/20k (expect XPROMPT_DEC_TAIL) =====" | tee -a "$LOG"
BE_SYS_LEN=20000 BE_FEWSHOT_LEN=20000 BS_KERNEL_TRACE_PLAN=1 \
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py $COMMON \
    --out "$RD/picker_20k.csv" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== (2) BS_KERNEL_NO_XPROMPT=1 @ 20k/20k (expect SHARED_2L_DEC_TAIL) =====" | tee -a "$LOG"
BE_SYS_LEN=20000 BE_FEWSHOT_LEN=20000 BS_KERNEL_NO_XPROMPT=1 BS_KERNEL_TRACE_PLAN=1 \
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py $COMMON \
    --out "$RD/picker_20k_noxp.csv" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "===== (3) unit tests =====" | tee -a "$LOG"
uv run python -m pytest tests/test_bs_kernel.py tests/test_bs_kernel_cache.py \
    -q 2>&1 | tail -40 | tee -a "$LOG"

echo "Done. $RD" | tee -a "$LOG"

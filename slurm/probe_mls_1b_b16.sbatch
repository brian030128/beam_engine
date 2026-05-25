#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=probe_mls_b16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/probe_mls_b16_%j.out
#SBATCH --error=slurm/logs/probe_mls_b16_%j.err

# One-off probe: multi_level_system at (B=16, K=64) on Llama-3.2-1B.
# In the K=64 exp1 sweep, B=4 (natural) gave 0/255 DEC_TAIL on this
# cell — too few prompts for the per-beam tail to outweigh root cost.
# Try B=16 to see if the picker swings here too.
#
# Single bench_sglang_e2e.py run with all baselines + the two forced
# 2L variants + bs_kernel picker, BS_KERNEL_TRACE_PLAN=1 so strategy
# mix prints inline. --repeat 1 --warmup.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B
export BS_KERNEL_TRACE_PLAN=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/probes"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/probe_mls_1b_b16.log"

{
    echo "=== probe multi_level_system (1B, B=16, K=64) started $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_level_system \
    --methods bs_kernel bsk_2l_1p bsk_2l_dt fasttree paged mlca \
    --paper-exact \
    --k-override 64 --b-override 16 \
    --max-new 256 --repeat 1 --warmup \
    --out "$OUT_DIR/probe_mls_1b_b16.csv" \
    2>&1 | tee -a "$LOG"

echo "Done. CSV: $OUT_DIR/probe_mls_1b_b16.csv" | tee -a "$LOG"
echo "Strategy mix inline in $LOG (search for 'strategy mix:')" | tee -a "$LOG"

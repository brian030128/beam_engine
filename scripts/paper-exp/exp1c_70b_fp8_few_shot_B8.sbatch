#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp1c_70b_fs_b8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/paper_exp1c_70b_fs_b8_%j.out
#SBATCH --error=slurm/logs/paper_exp1c_70b_fs_b8_%j.err

# Same 70B-fp8 multi_few_shot cell but B doubled 4 → 8 to see if
# the picker swings to DEC_TAIL when the per-rank KV slab gets
# heavier. May OOM at the prefill-stage cache allocation.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_KV_DTYPE=fp8_e4m3   # fp8 KV cache (halves the per-rank KV slab)
export BS_KERNEL_TRACE_PLAN=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/forced_2l_70b_fp8_B8"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/few_shot_B8.log"

{
    echo "=== exp1c 70B-FP8 few_shot B=8 started $(date -Iseconds) ==="
    echo "git HEAD: $(git rev-parse --short HEAD)"
    echo
} | tee "$LOG"

METHODS="bs_kernel bsk_2l_1p bsk_2l_dt"

echo "=== multi_few_shot (B=8, K=64) ===" | tee -a "$LOG"
torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods $METHODS --paper-exact \
    --scenarios multi_few_shot --k-override 64 --b-override 8 \
    --max-new 256 --repeat 1 --warmup \
    --out "$OUT_DIR/few_shot_B8.csv" \
    2>&1 | tee -a "$LOG"

echo "Done." | tee -a "$LOG"

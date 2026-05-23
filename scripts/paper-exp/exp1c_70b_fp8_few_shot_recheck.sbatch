#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp1c_70b_fs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_exp1c_70b_fs_%j.out
#SBATCH --error=slurm/logs/paper_exp1c_70b_fs_%j.err

# Re-check the Llama-3-70B-FP8 / multi_few_shot cell after cost-model
# update. Prior result: picker chose 100% 2L_1POOL, lost to forced
# 2L_DEC_TAIL by 9.5% (15,716 vs 14,358 ms). Question: does the new
# picker now select DEC_TAIL?

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
export BS_KERNEL_TRACE_PLAN=1   # print picker strategy histogram

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/forced_2l_70b_fp8_recheck"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/recheck.log"

{
    echo "=== exp1c 70B-FP8 few_shot recheck started $(date -Iseconds) ==="
    echo "git HEAD: $(git rev-parse --short HEAD)"
    echo "uncommitted cost_model.py changes:"
    git diff --stat src/beam_engine/methods/bs_kernel/cost_model.py 2>&1 | head
    echo
} | tee "$LOG"

METHODS="bs_kernel bsk_2l_1p bsk_2l_dt"

echo "=== multi_few_shot (B=4, K=64) ===" | tee -a "$LOG"
torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods $METHODS --paper-exact \
    --scenarios multi_few_shot --k-override 64 --b-override 4 \
    --max-new 256 --repeat 1 --warmup \
    --out "$OUT_DIR/few_shot.csv" \
    2>&1 | tee -a "$LOG"

echo "Done." | tee -a "$LOG"

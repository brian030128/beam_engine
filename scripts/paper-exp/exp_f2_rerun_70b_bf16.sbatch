#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f2_rerun_70b_bf16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/f2_rerun_70b_bf16_%j.out
#SBATCH --error=slurm/logs/f2_rerun_70b_bf16_%j.err

# Re-run F2 multi_few_shot (K=128 B=4) on 70B bf16 (TP=4) after the
# cost-model bw/compute-overlap fix, to check the picker now selects
# DEC_TAIL (was over-picking 1POOL 208/255 -> bs_kernel 16109 vs dt 14570).
# bs_kernel = picker; dt/1p forced refs (unchanged by the cost model).
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"; trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_WORKSPACE_MB=1024
export BE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16

OUT=benchmarks/bs_kernel/results/paper-exp/final_paper/Llama-3-70B-bf16
mkdir -p "$OUT" slurm/logs
LOG=$OUT/run_f2_rerun.log
echo "=== F2 rerun (cost-model fix) $(date -Iseconds) ===" | tee -a "$LOG"
torchrun --standalone --nproc_per_node=4 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods bs_kernel bsk_2l_dt bsk_2l_1p \
    --paper-exact --no-stage2 --max-new 256 --repeat 3 --warmup \
    --scenarios multi_few_shot --k-override 128 --b-override 4 \
    --out "$OUT/F2_multi_few_shot_K128_B4_costfix_mn256.csv" 2>&1 | tee -a "$LOG"
echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"

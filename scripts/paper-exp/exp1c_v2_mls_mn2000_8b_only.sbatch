#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_mls_mn2k_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/pe_mls_mn2k_8b_only_%j.out
#SBATCH --error=slurm/logs/pe_mls_mn2k_8b_only_%j.err

# 8B-only re-run of the multi_level_system max_new=2000 cell (Qwen3 already
# completed in 208576; 8B failed there due to a missing HF_TOKEN — the
# script set HF_HOME=/work but the gated 8B repo needed an online auth
# check). Fix here is HF_HUB_OFFLINE=1 so we use the cached weights.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export HF_HOME=/work/u4320956/hf
export HF_HUB_OFFLINE=1
export BE_MODEL="meta-llama/Llama-3.1-8B"

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1c_v2_mls_mn2000/Llama-3.1-8B"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== 8B multi_level_system K=64 B=4 max_new=2000 started $(date -Iseconds) ==="
    echo "model: $BE_MODEL  (tp=2)"
} | tee "$LOG"

torchrun --standalone --nproc_per_node=2 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods bs_kernel bsk_2l_1p bsk_2l_dt \
    --paper-exact --no-stage2 \
    --max-new 2000 --repeat 1 --warmup \
    --scenarios multi_level_system --k-override 64 --b-override 4 \
    --out "$OUT_DIR/multi_level_system.csv" \
    2>&1 | tee -a "$LOG"

echo "Done $(date -Iseconds)"

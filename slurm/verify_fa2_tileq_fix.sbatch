#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=tileqfix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/verify_fa2_tileq_fix_%j.out
#SBATCH --error=slurm/logs/verify_fa2_tileq_fix_%j.err

# Verify the FusedMultiLevelCascade adaptive_force_t16 fix (pad_frac_b gate):
#   - bsk_2l_dt fwd/step should drop 16.4 -> ~9.9 (CTA_TILE_Q 16 -> 64), matching mlca.
#   - bs_kernel (picker -> XPROMPT) should improve (its prefix uses the fused kernel).
#   - bsk_2l_1p UNAFFECTED (sets force_cta_tile_q -> adaptive_force_t16 gated off);
#     verify it still runs and isn't slower (the "don't break 1p" check).
# 1B FA2, K=16 B=16 sys=fewshot=20k, bf16.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16
export BE_FORCE_FA2=1

TS=$(date +%Y%m%d-%H%M%S)
RD="benchmarks/bs_kernel/results/verify_tileq_fix-${TS}"; mkdir -p "$RD"; LOG="$RD/bench.log"
COMMON="--no-stage2 --tree-root --max-new 96 --repeat 2 --warmup --scenarios multi_few_shot --k-override 16 --b-override 16"
echo "=== tile-Q fix verify (1B FA2) $(date -Iseconds) ===" | tee "$LOG"

for M in mlca bsk_2l_dt bsk_2l_1p bs_kernel; do
  echo "===== $M =====" | tee -a "$LOG"
  env BE_SYS_LEN=20000 BE_FEWSHOT_LEN=20000 BS_KERNEL_TRACE_PLAN=1 \
  torchrun --standalone --nproc_per_node=1 benchmarks/bs_kernel/bench_sglang_e2e.py \
      --methods $M $COMMON --out "$RD/$M.csv" 2>&1 | tee -a "$LOG"
done
echo "Done. $RD" | tee -a "$LOG"

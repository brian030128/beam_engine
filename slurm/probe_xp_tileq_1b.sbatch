#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp_tileq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/probe_xp_tileq_1b_%j.out
#SBATCH --error=slurm/logs/probe_xp_tileq_1b_%j.err

# Does bs_kernel's forced CTA_TILE_Q=128 pad the cross-prompt L1 bundle groups
# (K*gqa=64 rows -> 128) and explain the forward gap vs mlca (9.9 vs 13.7 ms)?
# Sweep BS_KERNEL_XP_TILE_Q for bsk_xp_dt: 128 (default), 64, 0 (auto per-level
# like mlca). mlca + bsk_2l_dt as references. 1B, K=16 B=16 sys=fewshot=20k,
# FA2, bf16.
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
RD="benchmarks/bs_kernel/results/probe_xp_tileq_1b-${TS}"; mkdir -p "$RD"; LOG="$RD/bench.log"
COMMON="--no-stage2 --tree-root --max-new 96 --repeat 2 --warmup --scenarios multi_few_shot --k-override 16 --b-override 16"
echo "=== XP_TILE_Q probe (1B, FA2) $(date -Iseconds) ===" | tee "$LOG"

echo "===== reference: mlca + bsk_2l_dt =====" | tee -a "$LOG"
env BE_SYS_LEN=20000 BE_FEWSHOT_LEN=20000 \
torchrun --standalone --nproc_per_node=1 benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods mlca bsk_2l_dt $COMMON --out "$RD/ref.csv" 2>&1 | tee -a "$LOG"

for TQ in 128 64 0; do
  echo "===== bsk_xp_dt  BS_KERNEL_XP_TILE_Q=$TQ =====" | tee -a "$LOG"
  env BE_SYS_LEN=20000 BE_FEWSHOT_LEN=20000 BS_KERNEL_XP_TILE_Q=$TQ \
  torchrun --standalone --nproc_per_node=1 benchmarks/bs_kernel/bench_sglang_e2e.py \
      --methods bsk_xp_dt $COMMON --out "$RD/xp_tq${TQ}.csv" 2>&1 | tee -a "$LOG"
done
echo "Done. $RD" | tee -a "$LOG"

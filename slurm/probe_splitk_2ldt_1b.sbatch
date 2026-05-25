#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=splitk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/probe_splitk_2ldt_1b_%j.out
#SBATCH --error=slurm/logs/probe_splitk_2ldt_1b_%j.err

# Hypothesis: bsk_2l_dt's fused L0 is 2x slower than mlca because the fused
# cascade's split-K gate (max_grid_size=2*num_sm=264) fires on L0 (128 CTAs <
# 264), splitting the 40k KV + paying merge_states overhead for ~no occupancy
# gain (128 is 97% of one 132-SM wave). Test: force L0 num_chunks=1 (no split)
# via FLASHINFER_FORCE_KV_CHUNKS="0:1" and see if bsk_2l_dt matches mlca.
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
RD="benchmarks/bs_kernel/results/probe_splitk_2ldt_1b-${TS}"; mkdir -p "$RD"; LOG="$RD/bench.log"
COMMON="--no-stage2 --tree-root --max-new 96 --repeat 2 --warmup --scenarios multi_few_shot --k-override 16 --b-override 16"
echo "=== split-K probe (1B FA2) $(date -Iseconds) ===" | tee "$LOG"

echo "===== mlca (reference) =====" | tee -a "$LOG"
env BE_SYS_LEN=20000 BE_FEWSHOT_LEN=20000 \
torchrun --standalone --nproc_per_node=1 benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods mlca $COMMON --out "$RD/mlca.csv" 2>&1 | tee -a "$LOG"

echo "===== bsk_2l_dt DEFAULT (split-K on) =====" | tee -a "$LOG"
env BE_SYS_LEN=20000 BE_FEWSHOT_LEN=20000 \
torchrun --standalone --nproc_per_node=1 benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods bsk_2l_dt $COMMON --out "$RD/2ldt_default.csv" 2>&1 | tee -a "$LOG"

echo "===== bsk_2l_dt + FLASHINFER_FORCE_KV_CHUNKS=0:1 (L0 no split) =====" | tee -a "$LOG"
env BE_SYS_LEN=20000 BE_FEWSHOT_LEN=20000 FLASHINFER_FORCE_KV_CHUNKS="0:1" \
torchrun --standalone --nproc_per_node=1 benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods bsk_2l_dt $COMMON --out "$RD/2ldt_nosplit.csv" 2>&1 | tee -a "$LOG"
echo "Done. $RD" | tee -a "$LOG"

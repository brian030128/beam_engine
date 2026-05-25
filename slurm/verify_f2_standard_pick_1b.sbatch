#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=f2pick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:35:00
#SBATCH --output=slurm/logs/verify_f2_standard_pick_1b_%j.out
#SBATCH --error=slurm/logs/verify_f2_standard_pick_1b_%j.err

# Standard F2 reset: multi_few_shot --paper-exact (NO tree-root), K=128 B=16, 1B.
# Confirm the bs_kernel picker selects SHARED_2L_DEC_TAIL (now that the fused
# CTA_TILE_Q bug is fixed) and that it ties/beats the baselines. Reference:
# bsk_2l_dt (forced), paged, mlca, fasttree. BE_FORCE_FA2 needed on 1B (hd=64).
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
RD="benchmarks/bs_kernel/results/verify_f2_standard-${TS}"; mkdir -p "$RD"; LOG="$RD/bench.log"
# Standard F2: --paper-exact, NO --tree-root, K=128 B=16, mn matches paper (256).
COMMON="--paper-exact --no-stage2 --max-new 128 --repeat 2 --warmup --scenarios multi_few_shot --k-override 128 --b-override 16"
echo "=== standard F2 picker check (1B, K=128 B=16, paper-exact) $(date -Iseconds) ===" | tee "$LOG"

echo "===== bs_kernel (picker) — expect SHARED_2L_DEC_TAIL =====" | tee -a "$LOG"
BS_KERNEL_TRACE_PLAN=1 torchrun --standalone --nproc_per_node=1 \
    benchmarks/bs_kernel/bench_sglang_e2e.py --methods bs_kernel $COMMON \
    --out "$RD/picker.csv" 2>&1 | tee -a "$LOG"

echo "===== references: bsk_2l_dt paged mlca fasttree =====" | tee -a "$LOG"
torchrun --standalone --nproc_per_node=1 \
    benchmarks/bs_kernel/bench_sglang_e2e.py --methods bsk_2l_dt paged mlca fasttree $COMMON \
    --out "$RD/refs.csv" 2>&1 | tee -a "$LOG"
echo "Done. $RD" | tee -a "$LOG"

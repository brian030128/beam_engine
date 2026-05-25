#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=f2_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/smoke_f2_full_1b_%j.out
#SBATCH --error=slurm/logs/smoke_f2_full_1b_%j.err

# Full F2 cell smoke on 1B at the NEW config (K=16, B=16, sys=fewshot=20k,
# --tree-root, NO --paper-exact) — mirrors run_sglang_attempt exactly. Runs the
# 1B SGLANG_CORE method set + forced bsk_xp_dt. Confirms: (a) every method runs
# at the long tree-root prefix (esp. fasttree/deft), (b) the bs_kernel picker
# selects XPROMPT_DEC_TAIL, (c) it beats the 2L baselines / matches forced
# bsk_xp_dt. max-new=128 (decision is the same as 256; faster).
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
export BE_KV_DTYPE=bf16

TS=$(date +%Y%m%d-%H%M%S)
RD="benchmarks/bs_kernel/results/smoke_f2_full_1b-${TS}"
mkdir -p "$RD"; LOG="$RD/bench.log"
echo "=== F2 full-cell smoke 1B (K=16 B=16 sys=fewshot=20k tree-root) $(date -Iseconds) ===" | tee "$LOG"

env BE_SYS_LEN=20000 BE_FEWSHOT_LEN=20000 BS_KERNEL_TRACE_PLAN=1 \
torchrun --standalone --nproc_per_node=1 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods paged fasttree deft mlca bs_kernel bsk_2l_1p bsk_2l_dt bsk_xp_dt \
    --no-stage2 --tree-root \
    --max-new 128 --repeat 2 --warmup \
    --scenarios multi_few_shot --k-override 16 --b-override 16 \
    --out "$RD/F2_smoke.csv" 2>&1 | tee -a "$LOG"
echo "Done. $RD" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=f2_fa2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/smoke_f2_fa2_1b_%j.out
#SBATCH --error=slurm/logs/smoke_f2_fa2_1b_%j.err

# Re-smoke the full F2 cell on 1B with BE_FORCE_FA2=1 (forces flashinfer
# prefill backend -> FA2, sidestepping the FA3/SM90 head_dim=64 compile bug).
# Expect: every method builds + runs (paged/fasttree/deft/mlca no longer hit
# the GMMA error), and the bs_kernel picker still selects XPROMPT_DEC_TAIL.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
export BE_DTYPE=bf16          # match KV dtype (real 1B driver sets both)
export BE_KV_DTYPE=bf16
export BE_FORCE_FA2=1          # the fix under test

TS=$(date +%Y%m%d-%H%M%S)
RD="benchmarks/bs_kernel/results/smoke_f2_fa2_1b-${TS}"; mkdir -p "$RD"; LOG="$RD/bench.log"
echo "=== F2 full-cell smoke 1B + FA2 (K=16 B=16 sys=fewshot=20k tree-root) $(date -Iseconds) ===" | tee "$LOG"

env BE_SYS_LEN=20000 BE_FEWSHOT_LEN=20000 BS_KERNEL_TRACE_PLAN=1 \
torchrun --standalone --nproc_per_node=1 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods paged fasttree deft mlca bs_kernel bsk_2l_dt bsk_xp_dt \
    --no-stage2 --tree-root \
    --max-new 96 --repeat 2 --warmup \
    --scenarios multi_few_shot --k-override 16 --b-override 16 \
    --out "$RD/F2_smoke.csv" 2>&1 | tee -a "$LOG"
echo "Done. $RD" | tee -a "$LOG"

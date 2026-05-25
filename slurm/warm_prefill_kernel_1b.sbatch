#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=warm_jit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/warm_prefill_kernel_1b_%j.out
#SBATCH --error=slurm/logs/warm_prefill_kernel_1b_%j.err

# The F2 full-cell smoke failed: the flashinfer SM90 head_dim=64 batch_prefill
# op (8 per-mask .cuda.o) failed to JIT-build (ninja code=1, nvcc stderr not
# surfaced) — the base .o built but every per-mask kernel is missing. Node has
# 527G RAM but the SLURM step set no --mem, so ~8 parallel nvcc likely OOM'd in
# the cgroup. Rebuild SERIALLY (MAX_JOBS=1) with --mem=120G to (a) confirm it's
# a parallel-build OOM (not a real compile error) and (b) warm the cache for
# the full run. paged alone triggers the full op build (needs mask_0/1).
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
export BE_KV_DTYPE=bf16
export MAX_JOBS=1          # serialize flashinfer JIT (torch cpp_extension)

TS=$(date +%Y%m%d-%H%M%S)
RD="benchmarks/bs_kernel/results/warm_jit_1b-${TS}"; mkdir -p "$RD"; LOG="$RD/bench.log"
echo "=== warm prefill kernel (serial JIT) $(date -Iseconds) ===" | tee "$LOG"

env BE_SYS_LEN=20000 BE_FEWSHOT_LEN=20000 \
torchrun --standalone --nproc_per_node=1 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods paged \
    --no-stage2 --tree-root \
    --max-new 16 --repeat 1 \
    --scenarios multi_few_shot --k-override 16 --b-override 16 \
    --out "$RD/warm.csv" 2>&1 | tee -a "$LOG"
echo "rc=$?  (if paged ran, the SM90 head_dim=64 prefill op now builds serially)" | tee -a "$LOG"
echo "Done. $RD" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp3l_8k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:35:00
#SBATCH --output=slurm/logs/bench_xprompt_3l_8k8k_%j.out
#SBATCH --error=slurm/logs/bench_xprompt_3l_8k8k_%j.err

# (1) 8k sys + 8k prompt timing: cross-prompt 3L vs 2L (B=16). Per-layer sys
#     KV = 16 MB (8192 tok); working set ~272 MB >> L2. Biggest read-saving case.
# (2) Probe ncu/nsys availability + perf-counter perms, and find the attention
#     kernel name, so a follow-up can profile DRAM reads to VERIFY sys is read
#     once (not B times).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL=${BE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
export BE_SYS_LEN=8192
export BE_FEWSHOT_LEN=8192

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/xprompt_3l_8k8k-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"
echo "=== 8k-sys + 8k-prompt $(date -Iseconds) ; $BE_MODEL ===" | tee "$LOG"

for B in 16; do
    echo "===== B=$B : 2L baseline (bsk_2l_dt) =====" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_few_shot --tree-root --methods bsk_2l_dt \
        --k-override 32 --b-override "$B" --max-new 128 --repeat 2 --warmup \
        --out "$RESULTS_DIR/B${B}_2l.csv" 2>&1 | tee -a "$LOG"

    echo "===== B=$B : cross-prompt 3L =====" | tee -a "$LOG"
    BS_KERNEL_CROSS_PROMPT_3L=1 \
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_few_shot --tree-root --methods bs_kernel \
        --k-override 32 --b-override "$B" --max-new 128 --repeat 2 --warmup \
        --out "$RESULTS_DIR/B${B}_xp3l.csv" 2>&1 | tee -a "$LOG"
done

echo "===== profiler probe =====" | tee -a "$LOG"
echo "-- which ncu/nsys --" | tee -a "$LOG"
which ncu nsys 2>&1 | tee -a "$LOG"
ncu --version 2>&1 | head -3 | tee -a "$LOG" || echo "ncu not found" | tee -a "$LOG"
echo "-- ncu perms probe on a trivial matmul --" | tee -a "$LOG"
ncu --metrics dram__bytes_read.sum --launch-count 1 --target-processes all \
    python -c "import torch; a=torch.randn(2048,2048,device='cuda'); (a@a).sum().item(); torch.cuda.synchronize()" \
    2>&1 | tail -25 | tee -a "$LOG" || echo "ncu probe failed (perms?)" | tee -a "$LOG"

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

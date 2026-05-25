#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp3l_4k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/bench_xprompt_3l_4k4k_%j.out
#SBATCH --error=slurm/logs/bench_xprompt_3l_4k4k_%j.err

# 4k sys + 4k prompt: bigger prefix (8192 vs 6656) => attention is a larger
# share of the forward. If the cross-prompt sys-dedup ever matters, a
# read-heavier prefix is where it should show. Cross-prompt 3L auto-detects
# the shared sys via page-ID LCA (no manual split boundary).
#   multi_few_shot, K=32, B in {4,16}, --tree-root, max_new=128
#   BE_SYS_LEN=4096 (256 pg), BE_FEWSHOT_LEN=4096 (256 pg)

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL=${BE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
export BE_SYS_LEN=4096
export BE_FEWSHOT_LEN=4096

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/xprompt_3l_4k4k-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"
echo "=== 4k-sys + 4k-prompt cross-prompt 3L vs 2L $(date -Iseconds) ; $BE_MODEL ===" | tee "$LOG"
echo "BE_SYS_LEN=$BE_SYS_LEN BE_FEWSHOT_LEN=$BE_FEWSHOT_LEN" | tee -a "$LOG"

for B in 4 16; do
    echo "===== B=$B : 2L baseline (bsk_2l_dt) =====" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_few_shot --tree-root --methods bsk_2l_dt \
        --k-override 32 --b-override "$B" \
        --max-new 128 --repeat 2 --warmup \
        --out "$RESULTS_DIR/B${B}_2l.csv" 2>&1 | tee -a "$LOG"

    echo "===== B=$B : cross-prompt 3L (bs_kernel + CROSS_PROMPT_3L) =====" | tee -a "$LOG"
    BS_KERNEL_CROSS_PROMPT_3L=1 \
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_few_shot --tree-root --methods bs_kernel \
        --k-override 32 --b-override "$B" \
        --max-new 128 --repeat 2 --warmup \
        --out "$RESULTS_DIR/B${B}_xp3l.csv" 2>&1 | tee -a "$LOG"
done
echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

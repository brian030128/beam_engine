#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=xp3l_bsw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/bench_xprompt_3l_bsweep_%j.out
#SBATCH --error=slurm/logs/bench_xprompt_3l_bsweep_%j.err

# Cross-prompt 3L vs 2L as B (number of prompts) grows. More prompts => more
# distinct L1 bundle groups => larger per-layer working set (sys + B*bundle).
# Once that exceeds the 50 MB H100 L2, the 2L baseline's B repeated sys reads
# stop being L2 hits and start hitting HBM — which is exactly where reading
# sys ONCE (cross-prompt L0 = single group of all B*K beams) should win.
#   multi_few_shot, K=32, B in {4,8,16,32}, --tree-root, max_new=128
#   sys=4096 tok (=8 MB/layer KV @1B), bundle=2560 tok (=5 MB/layer each)
#   => L2 (~50 MB) saturates around B~8-10.
# Trace OFF for clean timing (the XPROMPT path firing is already confirmed).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL=${BE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/xprompt_3l_bsweep-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"
echo "=== cross-prompt 3L B-sweep $(date -Iseconds) ; model $BE_MODEL ===" | tee "$LOG"

for B in 4 8 16 32; do
    echo "===== B=$B : 2L baseline (bsk_2l_dt) =====" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_few_shot --tree-root \
        --methods bsk_2l_dt \
        --k-override 32 --b-override "$B" \
        --max-new 128 --repeat 2 --warmup \
        --out "$RESULTS_DIR/B${B}_2l.csv" \
        2>&1 | tee -a "$LOG"

    echo "===== B=$B : cross-prompt 3L (bs_kernel + CROSS_PROMPT_3L) =====" | tee -a "$LOG"
    BS_KERNEL_CROSS_PROMPT_3L=1 \
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --scenarios multi_few_shot --tree-root \
        --methods bs_kernel \
        --k-override 32 --b-override "$B" \
        --max-new 128 --repeat 2 --warmup \
        --out "$RESULTS_DIR/B${B}_xp3l.csv" \
        2>&1 | tee -a "$LOG"
done

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"
echo "$RESULTS_DIR" > "$RESULTS_DIR/dir.txt"

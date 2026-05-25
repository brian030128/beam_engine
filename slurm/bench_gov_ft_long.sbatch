#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=gov_ft_l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/bench_gov_ft_long_%j.out
#SBATCH --error=slurm/logs/bench_gov_ft_long_%j.err

# fasttree + dbs_fasttree at the long-decode cell — apples-to-apples
# against the new bs_kernel numbers (post-DBS-topk-fix, post-suffix-
# aware cost-model). All four kernels share the page_driver topk path,
# so the DBS-topk fix benefits fasttree too — rerunning here so the
# comparison isn't biased by stale pre-fix numbers.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/gov_ft_long-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== gov_ft_long started $(date -Iseconds) ==="
    echo "cell:  L_p=8192, K=32, B=16, max_new=2048"
    echo
} | tee "$LOG"

for m in fasttree dbs_fasttree; do
    echo "=== running method: $m ===" | tee -a "$LOG"
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 32 --L_p 8192 --B 16 --max_new 2048 \
        --prompts-file data/gov_report.jsonl \
        --max_pages 80000 \
        --warmup \
        --methods "$m" \
        --out "$RESULTS_DIR/${m}.csv" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=3lv2l_cln
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/bench_3l_vs_2l_mfs_clean_%j.out
#SBATCH --error=slurm/logs/bench_3l_vs_2l_mfs_clean_%j.err

# Clean same-TreeSpec 2L-vs-3L forward isolation. Both runs use the
# multilevel TreeSpec (--tree-root) so the KV page layout is identical —
# removes the legacy-vs-multilevel confound from the first experiment.
#   Run C: good-2L  (no split knob)  → bsk_2l_dt = L0:sys+prompt | L2:beam
#   Run D: 3L       (split n=256)     → bsk_3l_dt = L0:sys | L1:prompt | L2:beam
# Separate invocations so the process-global split knob only touches Run D.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL=${BE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
export BS_KERNEL_TRACE_PLAN=1

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/3l_vs_2l_clean-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== clean 2L-vs-3L (same multilevel TreeSpec) $(date -Iseconds) ==="
    echo "model: $BE_MODEL ; cell: multi_few_shot K=32 B=4 max_new=256"
} | tee "$LOG"

echo "=== Run C: --tree-root, NO split → good-2L (bsk_2l_dt, bsk_2l_1p) ===" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_few_shot --tree-root \
    --methods bsk_2l_dt bsk_2l_1p \
    --k-override 32 --b-override 4 \
    --max-new 256 --repeat 3 --warmup \
    --out "$RESULTS_DIR/run_c_tree_2l.csv" \
    2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== Run D: --tree-root + split n=256 → 3L (bsk_3l_dt, bsk_3l_1p) ===" | tee -a "$LOG"
BS_KERNEL_FORCE_PREFIX_SPLIT_PAGES=256 \
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_few_shot --tree-root \
    --methods bsk_3l_dt bsk_3l_1p \
    --k-override 32 --b-override 4 \
    --max-new 256 --repeat 3 --warmup \
    --out "$RESULTS_DIR/run_d_tree_3l.csv" \
    2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"; echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

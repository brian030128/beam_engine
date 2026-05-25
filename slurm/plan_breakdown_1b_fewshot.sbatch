#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=plan_bd_fs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/plan_bd_fs_%j.out
#SBATCH --error=slurm/logs/plan_bd_fs_%j.err

# 1B / multi_few_shot (B=32, K=64, max_new=256), all five methods, one
# warmup + one measured pass. Trace flags on for fasttree + bs_kernel
# (so their dispatch_total_ms gets populated via the new
# bench_sglang_e2e column); MLCA_PLAN_CACHE=1 to match the paper's
# cache-on numbers.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export FT_TRACE_PLAN=1
export MLCA_PLAN_CACHE=1
export BE_MODEL=meta-llama/Llama-3.2-1B

OUT_DIR=benchmarks/bs_kernel/results/paper-exp/exp3_plan_breakdown/all_1b_fewshot
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== 1B/multi_few_shot plan-time breakdown $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "flags: BS_KERNEL_TRACE_PLAN=1 FT_TRACE_PLAN=1 MLCA_PLAN_CACHE=1"
    echo
} | tee "$LOG"

uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods paged fasttree mlca bs_kernel deft \
    --paper-exact \
    --max-new 256 --repeat 1 --warmup \
    --scenarios multi_few_shot --k-override 64 --b-override 32 \
    --out "$OUT_DIR/plan_breakdown.csv" \
    2>&1 | tee -a "$LOG"

echo
echo "=== plan-time breakdown table ===" | tee -a "$LOG"
python3 -c "
import csv
with open('$OUT_DIR/plan_breakdown.csv') as f:
    rows = list(csv.DictReader(f))
print(f'{\"method\":<10} {\"build_indices\":>14} {\"dispatch\":>10} {\"plan_total\":>11} {\"disp%\":>7}')
for r in rows:
    method = r['method']
    plan = float(r['plan_total_ms'])
    disp = float(r['dispatch_total_ms'])
    build = plan - disp
    pct = disp / plan * 100 if plan > 0 else 0
    print(f'{method:<10} {build:>14.2f} {disp:>10.2f} {plan:>11.2f} {pct:>6.1f}%')
" | tee -a "$LOG"

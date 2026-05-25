#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_verify10k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:30:00
#SBATCH --output=slurm/logs/verify_mn10000_1b_%j.out
#SBATCH --error=slurm/logs/verify_mn10000_1b_%j.err

# Verify at max_new=10000 (tails to 10k, past the ~4-5k 1POOL→DEC_TAIL
# crossover) on Llama-3.2-1B self_consistency K=16/B=1/L_p=32k:
#   A) is bs_kernel (the picker) still the fastest BASELINE? -> run the real
#      baselines paged/fasttree/deft/mlca alongside it.
#   B) does the picker select the RIGHT strategy per step? -> dump its per-step
#      strategy (lightweight, no cuda.sync) and compare its decode_total to the
#      forced 1pool / forced dt bounds and to the oracle step-schedule
#      (bsk_sched_1p_dt switching at the measured per-step crossover, step 4000).
#
# NOTE: BS_KERNEL_TRACE_PLAN is deliberately OFF — its per-step sync would
# inflate only bs_kernel's timing and bias check A. Per-step strategy comes
# from the no-sync _picked_strategy_log via BE_DUMP_STEP_TIMINGS instead.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME=/work/u4320956/hf
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
export BE_DTYPE=bf16            # 1B head_dim=64; FA3/SM90 build bug → force FA2
export BE_KV_DTYPE=bf16
export BE_FORCE_FA2=1
export BE_PICKER_SINGLE_LAUNCH=1   # 1POOL single-launch (matches prior runs)

export BE_SELF_CONSISTENCY_LP=32768
export BE_BSK_SCHED_SWITCH=4000    # oracle schedule: 1pool->dt at per-step crossover

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/final_paper/Llama-3.2-1B-bf16/sched"
STEP_DIR="$OUT_DIR/step_timings_mn10000"
mkdir -p "$STEP_DIR" slurm/logs
export BE_DUMP_STEP_TIMINGS="$STEP_DIR"
csv="$OUT_DIR/verify_pick_K16_B1_Lp32768_mn10000.csv"
rm -f "$csv"

# Recalibrate so the picker uses fresh bf16 coefficients (only bs_kernel cares;
# forced/scheduled ignore them).
uv run python -m beam_engine.methods.bs_kernel.calibrate --force 2>&1 | tail -n 3

torchrun --standalone --nproc_per_node=1 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods paged fasttree deft mlca bs_kernel \
              bsk_2l_1p_single bsk_2l_dt bsk_sched_1p_dt \
    --no-stage2 --max-new 10000 --repeat 2 --warmup \
    --scenarios self_consistency --k-override 16 --b-override 1 \
    --out "$csv"

echo "=== verify done $(date -Iseconds) ==="
echo "=== ANALYSIS ==="
uv run python scripts/paper-exp/analyze_pick_correctness.py "$csv" "$STEP_DIR" 2>&1 || true

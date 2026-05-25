#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_xover_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/xover_1b_%j.out
#SBATCH --error=slurm/logs/xover_1b_%j.err

# Probe the hypothesised 1POOL -> DEC_TAIL crossover on Llama-3.2-1B.
# Hypothesis: once the per-beam tail KV is long enough, DEC_TAIL's dedicated
# CTA_Q=1 decode kernel split-K's the tail into enough CTAs to saturate the
# SMs, while 1POOL's fused-cascade level-1 tail pool stays under-parallelised
# -> dt's per-step forward grows more slowly than 1pool's and eventually
# undercuts it. At max_new=4000 dt never won; here we push tails to 16k tokens
# and dump the FULL per-step forward stream for both forced strategies so the
# forward-vs-tail-length curves (and any crossover) can be read directly.
#
# Only the two forced strategies run (no picker / no schedule). Forcing makes
# the cost-model coefficients irrelevant, so no recalibration needed.
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
export BE_PICKER_SINGLE_LAUNCH=1   # 1POOL single-launch (matches prior F4 runs)

export BE_SELF_CONSISTENCY_LP=32768

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/final_paper/Llama-3.2-1B-bf16/sched"
STEP_DIR="$OUT_DIR/step_timings_mn16000"
mkdir -p "$STEP_DIR" slurm/logs
export BE_DUMP_STEP_TIMINGS="$STEP_DIR"
csv="$OUT_DIR/crossover_1p_vs_dt_K16_B1_Lp32768_mn16000.csv"
rm -f "$csv"

torchrun --standalone --nproc_per_node=1 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods bsk_2l_1p_single bsk_2l_dt \
    --no-stage2 --max-new 16000 --repeat 1 --warmup \
    --scenarios self_consistency --k-override 16 --b-override 1 \
    --out "$csv"

echo "=== crossover probe done $(date -Iseconds) ==="
echo "aggregate csv: $csv"
echo "per-step dumps: $STEP_DIR"

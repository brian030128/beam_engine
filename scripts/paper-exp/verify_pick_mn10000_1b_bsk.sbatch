#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_vfy10k_bsk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/verify_mn10000_1b_bsk_%j.out
#SBATCH --error=slurm/logs/verify_mn10000_1b_bsk_%j.err

# Parallel companion to verify_pick_mn10000_1b.sbatch: runs ONLY the
# bs_kernel-family methods (picker + forced 1p/dt + oracle schedule) so the
# picker-correctness check (B) lands fast instead of queuing behind the very
# slow deft baseline in the full job. Baselines (paged/fasttree/deft/mlca) for
# check A come from that other job. Separate STEP_DIR + CSV so the two jobs
# never clobber each other's dumps.
#
# No recalibration: the full job already wrote a fresh bf16 coefficient cache
# at its start, so we just load it (avoids a concurrent --force cache race).
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
STEP_DIR="$OUT_DIR/step_timings_mn10000_bsk"
mkdir -p "$STEP_DIR" slurm/logs
export BE_DUMP_STEP_TIMINGS="$STEP_DIR"
csv="$OUT_DIR/verify_pick_K16_B1_Lp32768_mn10000_bsk.csv"
rm -f "$csv"

torchrun --standalone --nproc_per_node=1 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods bs_kernel bsk_2l_1p_single bsk_2l_dt bsk_sched_1p_dt \
    --no-stage2 --max-new 10000 --repeat 2 --warmup \
    --scenarios self_consistency --k-override 16 --b-override 1 \
    --out "$csv"

echo "=== verify (bsk-only) done $(date -Iseconds) ==="
echo "=== ANALYSIS ==="
uv run python scripts/paper-exp/analyze_pick_correctness.py "$csv" "$STEP_DIR" 2>&1 || true

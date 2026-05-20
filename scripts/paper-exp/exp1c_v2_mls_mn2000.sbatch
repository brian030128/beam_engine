#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_mls_mn2k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/pe_mls_mn2k_%j.out
#SBATCH --error=slurm/logs/pe_mls_mn2k_%j.err

# multi_level_system at K=64, B=4, max_new=2000 — these are the cells
# where (a) FUSED is currently competitive or wins, and (b) the picker
# mixes FUSED + DECTAIL under the new cost model. Longer decode lets us
# see whether the mix decision pays off over the longer time horizon.
#
# Runs both 8B (TP=2 to fit max_new=2000 memory) and Qwen3-4B (TP=2).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export HF_HOME=/work/u4320956/hf
export HF_HUB_OFFLINE=1   # 8B is gated; rely on cached weights at /work

SGLANG_METHODS="bs_kernel bsk_2l_1p bsk_2l_dt"

run_cell () {
    local model_tag="$1"; shift
    local model="$1"; shift
    local out_dir="benchmarks/bs_kernel/results/paper-exp/exp1c_v2_mls_mn2000/${model_tag}"
    mkdir -p "$out_dir" slurm/logs
    local log="$out_dir/run.log"
    {
        echo "=== ${model_tag} multi_level_system K=64 B=4 max_new=2000 started $(date -Iseconds) ==="
        echo "model: $model"
    } | tee "$log"
    export BE_MODEL="$model"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $SGLANG_METHODS \
        --paper-exact --no-stage2 \
        --max-new 2000 --repeat 1 --warmup \
        --scenarios multi_level_system --k-override 64 --b-override 4 \
        --out "$out_dir/multi_level_system.csv" \
        2>&1 | tee -a "$log" \
        || echo "  (cell failed)" | tee -a "$log"
}

run_cell "Llama-3.1-8B" "meta-llama/Llama-3.1-8B"
run_cell "Qwen3-4B"     "Qwen/Qwen3-4B"

echo "Done $(date -Iseconds)"

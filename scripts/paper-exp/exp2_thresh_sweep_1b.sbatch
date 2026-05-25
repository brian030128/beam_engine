#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_thr_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/exp2_thr_sweep_1b_%j.out
#SBATCH --error=slurm/logs/exp2_thr_sweep_1b_%j.err
# Map the divergence-threshold survival boundary on the 1B gov_report cell.
# For each (children, tokens) gate, run bs_kernel_5ldt and report the depth
# distribution (strategy mix). Find the gentlest gate that still produces
# deep (3L/4L/5L) levels, then use it for the 8B run.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export HF_HOME=/work/u4320956/hf
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL=meta-llama/Llama-3.2-1B-Instruct
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16
export BE_FORCE_FA2=1
export BS_KERNEL_TRACE_PLAN=1
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp2_thr_sweep_1b"
mkdir -p "$OUT_DIR" slurm/logs
# combos: "children tokens"
for combo in "1 0" "1 16" "1 32" "2 0" "2 16"; do
    set -- $combo; ch=$1; tok=$2
    echo "########## GATE children>${ch} tokens>${tok} ##########"
    BS_KERNEL_MIN_LEVEL_CHILDREN=$ch BS_KERNEL_MIN_LEVEL_TOKENS=$tok \
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 32 --L_p 8192 --B 8 --max_new 2048 \
        --prompts-file data/gov_report.jsonl --max_pages 40000 --warmup \
        --measure_start 1500 --measure_end 2000 \
        --methods bs_kernel_5ldt --out "$OUT_DIR/ch${ch}_tok${tok}.csv" 2>&1
    echo
done
echo "Done sweep."

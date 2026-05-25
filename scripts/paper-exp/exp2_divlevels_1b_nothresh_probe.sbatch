#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_div1b_nothr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/exp2_div1b_nothresh_%j.out
#SBATCH --error=slurm/logs/exp2_div1b_nothresh_%j.err
# Control probe: same 1B gov_report cell, NO divergence threshold, to show
# the natural depth distribution (how often beams reach 3L/4L/5L without
# the >128tok/>8child gate). Compare strategy mix vs the gated run.
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
# NOTE: threshold env intentionally UNSET (defaults = split on any fork).
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp2_divlevels_1b_nothresh"
mkdir -p "$OUT_DIR" slurm/logs
for m in bs_kernel_3ldt bs_kernel_5l1p bs_kernel_5ldt; do
    echo "=== running method: $m (no threshold) ==="
    uv run python benchmarks/bs_kernel/bench_batched.py \
        --K 32 --L_p 8192 --B 8 --max_new 2048 \
        --prompts-file data/gov_report.jsonl --max_pages 40000 --warmup \
        --measure_start 1500 --measure_end 2000 \
        --methods "$m" --out "$OUT_DIR/${m}.csv" 2>&1
    echo
done
echo "Done. Results: $OUT_DIR"

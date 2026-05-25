#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=smoke_multidraw_bsk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/smoke_multidraw_bsk_%j.out
#SBATCH --error=slurm/logs/smoke_multidraw_bsk_%j.err

# Smoke test for the multi-draw data-seed axis (bs_kernel only, 1B, fast).
# Validates: data_seed threading, shape-conform across draws, data_seed CSV
# column, warmup-once. NOT a perf run — tiny K/B/L_p/max_new.
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
export BS_KERNEL_TRACE_PLAN=1
export BE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16
export BE_FORCE_FA2=1                 # 1B hd=64: FA3 batch_prefill won't build
export BE_SELF_CONSISTENCY_LP=4096    # small prefix so F4 path is quick
export BE_HOTPOTQA_MULTI_Q_PATH="data/hotpotqa_multi_q_K128_80k.jsonl"

OUT="benchmarks/bs_kernel/results/smoke_multidraw_bsk.csv"
echo "=== multi-draw smoke (bs_kernel only) $(date -Iseconds) ==="
echo "calibrating..."
uv run python -m beam_engine.methods.bs_kernel.calibrate --force 2>&1 | tail -3

# multi_few_shot exercises the conform path (its shared length varies across
# draws); self_consistency exercises the (already shape-pinned) F4 path.
# K=16 B=2, max_new=16, two data draws.
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods bs_kernel \
    --paper-exact --no-stage2 \
    --max-new 16 --warmup --data-seeds 0,1 \
    --scenarios multi_few_shot self_consistency \
    --k-override 16 --b-override 2 \
    --out "$OUT"

echo
echo "=== CSV head (expect data_seed column; seeds 0 and 1 per scenario) ==="
cut -d, -f1-5 "$OUT"
echo "=== done $(date -Iseconds) ==="

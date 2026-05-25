#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_val_paged
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/val_paged_%j.out
#SBATCH --error=slurm/logs/val_paged_%j.err

# Validate the paged LCA-prefix index-build optimization:
#   1) correctness — paged still token-equal to tree (tests/test_baselines.py)
#   2) plan-time   — paged plan_total on 8B F1 (was ~3136 ms) should drop sharply
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
export BE_HOTPOTQA_MULTI_Q_PATH=data/hotpotqa_multi_q_K128_80k.jsonl

OUT=benchmarks/bs_kernel/results/paper-exp/_validate_paged_cache
mkdir -p "$OUT" slurm/logs

echo "===== 1) CORRECTNESS: paged vs tree token equality ====="
uv run python tests/test_baselines.py 2>&1 | tee "$OUT/equality.log"

echo "===== 2) PLAN-TIME: paged on 8B F1 multi_doc_qa K=128 B=1 ====="
export BE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16
export PAGED_TRACE_PLAN=1
torchrun --standalone --nproc_per_node=1 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods paged --no-stage2 --max-new 256 --repeat 3 --warmup \
    --scenarios multi_doc_qa --k-override 128 --b-override 1 \
    --out "$OUT/paged_8b_F1.csv" 2>&1 | tee "$OUT/paged_8b_F1.log"

echo "===== paged plan_total_ms (new) ====="
uv run python -c "
import csv,statistics as s
v=[float(r['plan_total_ms']) for r in csv.DictReader(open('$OUT/paged_8b_F1.csv')) if r['method']=='paged']
d=[float(r['decode_total_ms']) for r in csv.DictReader(open('$OUT/paged_8b_F1.csv')) if r['method']=='paged']
print(f'paged 8B F1: plan_total_ms={s.median(v):.0f} (was 3136)  decode_total_ms={s.median(d):.0f} (was 47384)')
"
echo "=== done $(date -Iseconds) ==="

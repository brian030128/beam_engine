#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=smoke_deft_cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/smoke_deft_cache_%j.out
#SBATCH --error=slurm/logs/smoke_deft_cache_%j.err

# A/B test: DeftBackend with the LCA cache on (BE_DEFT_PLAN_CACHE=1,
# default) vs off (BE_DEFT_PLAN_CACHE=0). Three cells exercising
# different page-period / fork dynamics:
#   small : K=4,  L_p=128,  B=2,  max_new=64
#   midK  : K=64, L_p=512,  B=2,  max_new=64    (BLOCK_M chunking)
#   long  : K=32, L_p=1024, B=4,  max_new=512   (longer page periods)
#
# Compares decode_total_ms / plan_total_ms across (cache_on, cache_off).
# Cache hit rate is printed via BE_DEFT_PRINT_CACHE_STATS=1 (handled
# below via a small wrapper). If the cache is correct, decode_total_ms
# should match within timing noise across runs; plan_total_ms should
# shrink with cache on. Also runs fasttree for a sanity reference.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR=benchmarks/bs_kernel/results/smoke
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/deft_cache_smoke.log"

{
    echo "=== deft cache A/B smoke $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

run_cell () {
    local tag="$1"; shift
    local K="$1"; shift
    local L_p="$1"; shift
    local B="$1"; shift
    local maxnew="$1"; shift
    local methods="$1"; shift
    local cache="$1"; shift

    {
        echo "--- cell ${tag}: K=${K} L_p=${L_p} B=${B} max_new=${maxnew} ---"
        echo "    methods=${methods}  BE_DEFT_PLAN_CACHE=${cache}"
    } | tee -a "$LOG"

    BE_DEFT_PLAN_CACHE=$cache uv run python benchmarks/bs_kernel/bench_batched.py \
        --K $K --L_p $L_p --B $B --max_new $maxnew \
        --methods $methods \
        --warmup \
        --out "$OUT_DIR/deft_cache_${tag}_c${cache}.csv" \
        2>&1 | tee -a "$LOG"
}

# small cell — covers single-chunk leaves, frequent page boundaries
# (max_new=64 = 4 page periods)
run_cell "small"  4   128  2  64  "deft fasttree"  1
run_cell "small"  4   128  2  64  "deft"           0

# midK cell — covers BLOCK_M chunking, K=64 readers on per-prompt root
run_cell "midk"   64  512  2  64  "deft fasttree"  1
run_cell "midk"   64  512  2  64  "deft"           0

# long cell — long page periods so cache amortisation matters
run_cell "long"   32  1024 4  512 "deft fasttree"  1
run_cell "long"   32  1024 4  512 "deft"           0

echo "Done." | tee -a "$LOG"

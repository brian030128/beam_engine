#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=ft_tp_fix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/verify_fasttree_tp_fix_%j.out
#SBATCH --error=slurm/logs/verify_fasttree_tp_fix_%j.err

# Verifies the fasttree TP fix (use global num_kv_heads in the
# parallelism heuristic).
#   1. TP=2 fasttree on the long-decode prompt - should now match TP=1's
#      score (-174.7993).
#   2. TP=2 smoke test with fasttree backend - cross-rank tokens identical.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== fasttree TP fix verification started $(date -Iseconds) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo

echo "############ Step 1: TP=2 fasttree long-decode (expect score ~ -174.80) ############"
torchrun --standalone --nproc_per_node=2 \
    scripts/run_tp_beam_search.py \
    --model meta-llama/Llama-3.1-8B \
    --backend fasttree \
    --beam-width 32 \
    --max-new 2048 \
    --dtype fp16 \
    || echo "  (TP=2 fasttree failed)"

echo
echo "############ Step 2: TP=2 smoke test, fasttree backend (cross-rank token check) ############"
BE_SMOKE_BACKENDS=fasttree torchrun --standalone --nproc_per_node=2 tests/test_tp_smoke.py \
    || echo "  (smoke test failed)"

echo
echo "=== done $(date -Iseconds) ==="

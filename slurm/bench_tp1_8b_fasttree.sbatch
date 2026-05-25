#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=tp1_ft_8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/tp1_8b_fasttree_%j.out
#SBATCH --error=slurm/logs/tp1_8b_fasttree_%j.err

# TP=1 baseline for fasttree on Llama-3.1-8B with the same long-decode
# prompt used by tp2 — to determine whether the score divergence vs
# bs_kernel/paged is TP-specific or just fasttree-vs-paged numerics.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TP=1 8B fasttree baseline started $(date -Iseconds) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

for backend in fasttree paged bs_kernel; do
    echo "############ backend=${backend} (TP=1) ############"
    python scripts/run_tp_beam_search.py \
        --model meta-llama/Llama-3.1-8B \
        --backend "${backend}" \
        --beam-width 32 \
        --max-new 2048 \
        --dtype fp16 \
        || echo "  (${backend} failed; continuing)"
    echo
done

echo "=== done $(date -Iseconds) ==="

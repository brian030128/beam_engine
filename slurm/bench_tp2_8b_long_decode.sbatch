#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=tp2_8b_lng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/tp2_8b_long_decode_%j.out
#SBATCH --error=slurm/logs/tp2_8b_long_decode_%j.err

# Llama-3.1-8B beam search across 2 H100s via tensor parallel.
# Uses scripts/run_tp_beam_search.py (the only TP-aware entrypoint —
# bench_sglang_e2e.py / bench_batched.py do not yet thread init_tp).
# Long-decode setting: max_new=2048, beam_width=32.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TP=2 8B long-decode beam search started $(date -Iseconds) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo

for backend in bs_kernel paged fasttree; do
    echo "############ backend=${backend} ############"
    t0=$(date +%s)
    torchrun --standalone --nproc_per_node=2 \
        scripts/run_tp_beam_search.py \
        --model meta-llama/Llama-3.1-8B \
        --backend "${backend}" \
        --beam-width 32 \
        --max-new 2048 \
        --dtype fp16 \
        || echo "  (${backend} failed; continuing)"
    t1=$(date +%s)
    echo "  wall=$((t1 - t0))s"
    echo
done

echo "=== done $(date -Iseconds) ==="

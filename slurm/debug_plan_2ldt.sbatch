#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=dbgplan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/debug_plan_2ldt_%j.out
#SBATCH --error=slurm/logs/debug_plan_2ldt_%j.err

# Print the fused-cascade plan's actual gqa / max_packed / large_cta_tile_q for
# bsk_2l_dt's dec_tail prefix (expect gqa=4, max_packed=64, large_cta_tile_q=64;
# if it's gqa=1/max_packed=16/CTA=16 that's the bug). max-new=2, 1 repeat.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
export BE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16
export BE_FORCE_FA2=1
export FLASHINFER_DEBUG_PLAN=1

RD="benchmarks/bs_kernel/results/debug_plan_2ldt-$(date +%H%M%S)"; mkdir -p "$RD"
env BE_SYS_LEN=20000 BE_FEWSHOT_LEN=20000 \
torchrun --standalone --nproc_per_node=1 benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods bsk_2l_dt --no-stage2 --tree-root --max-new 2 --repeat 1 \
    --scenarios multi_few_shot --k-override 16 --b-override 16 \
    --out "$RD/x.csv" 2>&1 | grep -E "FI_DEBUG_PLAN" | sort -u | head -20
echo "Done."

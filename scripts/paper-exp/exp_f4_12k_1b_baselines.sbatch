#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f4_12k_1b_bl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/f4_12k_1b_bl_%j.out
#SBATCH --error=slurm/logs/f4_12k_1b_bl_%j.err

# Baseline companion to exp_f4_12k_1b.sbatch: mlca/fasttree/deft only, written
# to a SEPARATE *.baselines.csv so it doesn't clobber the main job's CSV. Same
# paper F4 cell (1B bf16 TP=1, K=16 B=1 L_p=32K, max_new=12000). deft is slow;
# the bench now writes incrementally so mlca/fasttree survive a deft timeout.
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
export BE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16
export BE_FORCE_FA2=1                 # 1B hd=64 FA3 build bug (affects fasttree/mlca prefill)
export BE_SELF_CONSISTENCY_LP=32768

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/final_paper/Llama-3.2-1B-bf16"
mkdir -p "$OUT_DIR" slurm/logs
csv="$OUT_DIR/F4_self_consistency_K16_B1_Lp32768_mn12000.baselines.csv"

torchrun --standalone --nproc_per_node=1 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods mlca fasttree deft \
    --no-stage2 --max-new 12000 --repeat 2 --warmup \
    --scenarios self_consistency --k-override 16 --b-override 1 \
    --out "$csv"
echo "=== 1B baselines done $(date -Iseconds): $csv ==="

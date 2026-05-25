#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f4_12k_70b_bl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H100:4
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/f4_12k_70b_bl_%j.out
#SBATCH --error=slurm/logs/f4_12k_70b_bl_%j.err

# Baseline companion to exp_f4_12k_70b_bf16.sbatch: mlca/fasttree/deft only ->
# separate *.baselines.csv. Paper F4 cell (70B bf16 TP=4, K=16 B=1 L_p=64K,
# max_new=12000). repeat=1 (deft is very slow at this scale); methods ordered
# mlca,fasttree,deft so the incremental CSV write captures mlca/fasttree even
# if deft runs past the wall clock.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"; trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_WORKSPACE_MB=2048
export BE_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16
export BE_SELF_CONSISTENCY_LP=65536

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/final_paper/Llama-3-70B-bf16"
mkdir -p "$OUT_DIR" slurm/logs
csv="$OUT_DIR/F4_self_consistency_K16_B1_Lp65536_mn12000.baselines.csv"

torchrun --standalone --nproc_per_node=4 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods mlca fasttree deft \
    --no-stage2 --max-new 12000 --repeat 1 --warmup \
    --scenarios self_consistency --k-override 16 --b-override 1 \
    --out "$csv"
echo "=== 70B baselines done $(date -Iseconds): $csv ==="

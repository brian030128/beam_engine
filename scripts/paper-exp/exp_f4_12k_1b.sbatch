#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f4_12k_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/f4_12k_1b_%j.out
#SBATCH --error=slurm/logs/f4_12k_1b_%j.err

# Final-paper F4 (self_consistency) cell — Llama-3.2-1B (bf16, TP=1) — but
# with a LONG decode: max_new=12000 (vs 256 in the headline cell). Method set
# trimmed to the picker + paged + the two forced bs_kernel strategies
# (fasttree/deft/mlca skipped). BS_KERNEL_TRACE_PLAN deliberately OFF so the
# per-step cuda.sync doesn't bias bs_kernel's timing vs paged; the picker's
# per-step strategy comes from the no-sync dump (BE_DUMP_STEP_TIMINGS).
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
export BE_DTYPE=bf16            # 1B head_dim=64; FA3/SM90 build bug → force FA2
export BE_KV_DTYPE=bf16
export BE_FORCE_FA2=1

export F4_MN=12000
export F4_REPEAT=2
export F4_LP=32768             # paper 1B F4 prefix length (build_final_results.py)

MODEL_TAG="Llama-3.2-1B-bf16"
NPROC=1
export BE_DUMP_STEP_TIMINGS="benchmarks/bs_kernel/results/paper-exp/final_paper/${MODEL_TAG}/step_timings_mn12000"
SC_CORE="paged bs_kernel bsk_2l_1p_single bsk_2l_dt"
SC_EXTRA=""
source scripts/paper-exp/final_f4_lib.sh

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_mdqa_b2_40k_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/mdqa_b2_40k_1b_%j.out
#SBATCH --error=slurm/logs/mdqa_b2_40k_1b_%j.err

# multi_doc_qa K=128 B=2 L_p=40K — Llama-3.2-1B-bf16 (bf16). Matches the 70B B=4/40K cell.
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
export BE_WORKSPACE_MB=512
export BE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
export BE_DTYPE=bf16
export BE_KV_DTYPE=bf16
export BE_HOTPOTQA_MULTI_Q_PATH=data/hotpotqa_multi_q_K128_40k.jsonl

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/final_paper/Llama-3.2-1B-bf16"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run_mdqa_b2_40k.log"
csv="$OUT_DIR/F1c_multi_doc_qa_K128_B2_Lp40k_mn256.csv"
echo "=== multi_doc_qa K=128 B=2 L_p=40K Llama-3.2-1B-bf16 $(date -Iseconds) ===" | tee -a "$LOG"

torchrun --standalone --nproc_per_node=1 \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods paged fasttree deft mlca bs_kernel bsk_2l_1p bsk_2l_dt \
    --no-stage2 --max-new 256 --repeat 3 --warmup \
    --scenarios multi_doc_qa --k-override 128 --b-override 2 \
    --out "$csv" 2>&1 | tee -a "$LOG"

echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"

#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_recheck_1b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/recheck_1b_%j.out
#SBATCH --error=slurm/logs/recheck_1b_%j.err
# Regression check: after the split-K cost-model change, does bs_kernel still
# pick the right family per cell? Expect DEC_TAIL on F1/F2/F3, 1POOL on F4.
set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh; module load cuda/12.4; source .venv/bin/activate
export HF_HOME=/work/u4320956/hf
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BS_KERNEL_TRACE_PLAN=1
export BE_WORKSPACE_MB=512
export BE_MODEL="meta-llama/Llama-3.2-1B-Instruct"; export BE_DTYPE=bf16; export BE_KV_DTYPE=bf16
OUT=benchmarks/bs_kernel/results/paper-exp/recheck/Llama-3.2-1B-bf16; mkdir -p "$OUT" slurm/logs
LOG=$OUT/run.log
echo "=== picker recheck Llama-3.2-1B-bf16 $(date -Iseconds) ===" | tee -a "$LOG"
# F1 multi_doc_qa K=128 B=1 (80k prefix)
BE_HOTPOTQA_MULTI_Q_PATH=data/hotpotqa_multi_q_K128_80k.jsonl \
torchrun --standalone --nproc_per_node=1 benchmarks/bs_kernel/bench_sglang_e2e.py \
  --methods bs_kernel --no-stage2 --max-new 256 --repeat 1 --warmup \
  --scenarios multi_doc_qa --k-override 128 --b-override 1 --out "$OUT/F1.csv" 2>&1 | tee -a "$LOG"
# F2 multi_few_shot K=128
torchrun --standalone --nproc_per_node=1 benchmarks/bs_kernel/bench_sglang_e2e.py \
  --methods bs_kernel --paper-exact --no-stage2 --max-new 256 --repeat 1 --warmup \
  --scenarios multi_few_shot --k-override 128 --b-override 16 --out "$OUT/F2.csv" 2>&1 | tee -a "$LOG"
# F3 beam_search K=64 Lp16000 (batched)
torchrun --standalone --nproc_per_node=1 benchmarks/bs_kernel/bench_batched.py \
  --K 64 --L_p 16000 --B 16 --max_new 256 --prompts-file data/hotpotqa.jsonl \
  --warmup --methods bs_kernel --out "$OUT/F3.csv" 2>&1 | tee -a "$LOG"
# F4 self_consistency K=16 B=1 Lp16384 (single-launch picker)
BE_SELF_CONSISTENCY_LP=16384 BE_PICKER_SINGLE_LAUNCH=1 \
torchrun --standalone --nproc_per_node=1 benchmarks/bs_kernel/bench_sglang_e2e.py \
  --methods bs_kernel --no-stage2 --max-new 256 --repeat 1 --warmup \
  --scenarios self_consistency --k-override 16 --b-override 1 --out "$OUT/F4.csv" 2>&1 | tee -a "$LOG"
echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"

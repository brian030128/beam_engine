#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=3lv2l_mfs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:25:00
#SBATCH --output=slurm/logs/bench_3l_vs_2l_mfs_%j.out
#SBATCH --error=slurm/logs/bench_3l_vs_2l_mfs_%j.err

# Does a 3-level decode cascade (L0=sys | L1=prompt | L2=K-beam tails) beat
# the merged 2L on multi_few_shot? See plan
# .claude/plans/git-state-is-coherent-federated-hare.md
#
#   cell: multi_few_shot, K=32, B=4 (128 leaves), max_new=256
#   sys=4096 tok=256 pages, fewshot bundle=2560 tok=160 pages, page_size=16
#   → force the sys|prompt split at n=256 pages.
#
# Run A (baseline): legacy 2-level TreeSpec (no --tree-root, no split knob).
#   bsk_2l_* give the explicit old-2L (L0=sys+prompt, L1=K beams) timings;
#   bs_kernel confirms the legacy picker also stays 2L.
# Run B (3L): multilevel TreeSpec (--tree-root) + BS_KERNEL_FORCE_PREFIX_SPLIT_PAGES=256.
#   bsk_3l_* force the genuine 3L (L0=sys, L1=prompt, L2=K beams); bs_kernel
#   here sees BOTH 2L and 3L candidates → answers "does the picker pick 3L?".
# BS_KERNEL_TRACE_PLAN=1 logs the per-step chosen strategy + dispatch depth so
# we can confirm 3L actually dispatched (not the silent collapse-to-2L).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/work/u4320956/hf
# 1B-Instruct (cached on /work; the HF token at ~/.cache/huggingface/token
# grants access — that's why it's cached). The model loader
# (modeling_llama.from_pretrained) calls snapshot_download(repo_id), so a
# repo-id is required (not a local path). Leave HF_HUB_OFFLINE unset:
# offline triggers a transformers tokenizer crash (_patch_mistral_regex →
# model_info). Same arch/shapes as base 1B → decode timing is comparable
# for this 3L-vs-2L comparison.
export BE_MODEL=${BE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
export BS_KERNEL_TRACE_PLAN=1

TS=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="benchmarks/bs_kernel/results/3l_vs_2l_mfs-${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/bench.log"

{
    echo "=== 3L-vs-2L multi_few_shot started $(date -Iseconds) ==="
    echo "model: $BE_MODEL"
    echo "cell:  multi_few_shot K=32 B=4 max_new=256; split n=256 (sys boundary)"
    echo
} | tee "$LOG"

echo "=== Run A: legacy old-2L baseline (no --tree-root, no split) ===" | tee -a "$LOG"
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_few_shot \
    --methods bs_kernel bsk_2l_dt bsk_2l_1p \
    --k-override 32 --b-override 4 \
    --max-new 256 --repeat 3 --warmup \
    --out "$RESULTS_DIR/run_a_legacy_2l.csv" \
    2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== Run B: multilevel + forced 3L (--tree-root, split n=256) ===" | tee -a "$LOG"
BS_KERNEL_FORCE_PREFIX_SPLIT_PAGES=256 \
uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
    --scenarios multi_few_shot --tree-root \
    --methods bs_kernel bsk_3l_1p bsk_3l_dt \
    --k-override 32 --b-override 4 \
    --max-new 256 --repeat 3 --warmup \
    --out "$RESULTS_DIR/run_b_tree_3l.csv" \
    2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "Done. Results: $RESULTS_DIR" | tee -a "$LOG"

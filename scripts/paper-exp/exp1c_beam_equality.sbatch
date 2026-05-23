#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp1c_beams
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_exp1c_beams_%j.out
#SBATCH --error=slurm/logs/paper_exp1c_beams_%j.err

# Beam-equality validation across picker / forced 2L_1POOL / forced
# 2L_DEC_TAIL on the same cells as exp1c. max_new shortened to 64
# (sufficient to catch any divergence — methods that diverge do so
# within ~10-20 tokens) so we can cover 1B/8B/Qwen3 + 70B without
# blowing the dev-partition cap.
#
# Dumps each method's leaf_token_ids per cell to JSON via
# BE_DUMP_BEAMS_DIR; a follow-up Python step compares them.
#
# Output: benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/beam_equality/

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 70B FP8 weights cache to SLURM_TMPDIR (ephemeral) per memory note.
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT
export HF_TOKEN="${HF_TOKEN:-}"

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/beam_equality"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/beam_equality.log"
export BE_DUMP_BEAMS_DIR="$OUT_DIR"
METHODS="bs_kernel bsk_2l_1p bsk_2l_dt"

{
    echo "=== exp1c beam-equality run started $(date -Iseconds) ==="
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo
} | tee "$LOG"

# --- Single-GPU models ---
for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    echo "############ ${model_tag} ############" | tee -a "$LOG"
    export BE_MODEL="meta-llama/${model_tag}"
    export BE_DTYPE=fp16
    # Sub-dir per model so dumps don't collide.
    export BE_DUMP_BEAMS_DIR="$OUT_DIR/${model_tag}"
    mkdir -p "$BE_DUMP_BEAMS_DIR"
    # 1B uses larger B on chain/few_shot per exp1c; 8B uses B=8.
    if [[ "$model_tag" == "Llama-3.2-1B" ]]; then
        B_CHAIN=32; B_FEW=32
    else
        B_CHAIN=8;  B_FEW=8
    fi
    for cell in "multi_level_system 4" "multi_chain_reasoning $B_CHAIN" "multi_few_shot $B_FEW"; do
        read scenario B <<< "$cell"
        echo "=== $scenario B=$B K=64 max_new=64 ===" | tee -a "$LOG"
        uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
            --methods $METHODS --paper-exact \
            --scenarios "$scenario" --k-override 64 --b-override "$B" \
            --max-new 64 --repeat 1 \
            --out "$OUT_DIR/${model_tag}_${scenario}.csv" \
            2>&1 | tee -a "$LOG" || true
    done
done

# --- Qwen3-4B (TP=2) ---
echo "############ Qwen3-4B TP=2 ############" | tee -a "$LOG"
export BE_MODEL="Qwen/Qwen3-4B"
export BE_DTYPE=fp16
export BE_DUMP_BEAMS_DIR="$OUT_DIR/Qwen3-4B"
mkdir -p "$BE_DUMP_BEAMS_DIR"
for cell in "multi_level_system 4" "multi_chain_reasoning 8" "multi_few_shot 8"; do
    read scenario B <<< "$cell"
    echo "=== Qwen3-4B $scenario B=$B K=64 max_new=64 ===" | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS --paper-exact \
        --scenarios "$scenario" --k-override 64 --b-override "$B" \
        --max-new 64 --repeat 1 \
        --out "$OUT_DIR/Qwen3-4B_${scenario}.csv" \
        2>&1 | tee -a "$LOG" || true
done

# --- Llama-3-70B-FP8 (TP=2) ---
echo "############ Llama-3-70B-FP8 TP=2 ############" | tee -a "$LOG"
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BE_DUMP_BEAMS_DIR="$OUT_DIR/Llama-3-70B-FP8"
mkdir -p "$BE_DUMP_BEAMS_DIR"
for cell in "multi_level_system 4" "multi_chain_reasoning 4" "multi_few_shot 4"; do
    read scenario B <<< "$cell"
    echo "=== 70B-FP8 $scenario B=$B K=64 max_new=64 ===" | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS --paper-exact \
        --scenarios "$scenario" --k-override 64 --b-override "$B" \
        --max-new 64 --repeat 1 \
        --out "$OUT_DIR/Llama-3-70B-FP8_${scenario}.csv" \
        2>&1 | tee -a "$LOG" || true
done

echo "Done. Beam dumps in $OUT_DIR/<model>/<scenario>__<method>.json" | tee -a "$LOG"

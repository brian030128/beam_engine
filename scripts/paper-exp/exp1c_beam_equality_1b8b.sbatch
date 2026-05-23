#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp1c_beams_1b8b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/paper_exp1c_beams_1b8b_%j.out
#SBATCH --error=slurm/logs/paper_exp1c_beams_1b8b_%j.err

# Beam-equality validation for 1B + 8B (the 70B/Qwen3 dumps from
# the first run succeeded; 1B/8B failed because the prior sbatch
# overrode HF_HOME to SLURM_TMPDIR which doesn't have the gated
# meta-llama configs cached).

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Use default HF cache at ~/.cache/huggingface where meta-llama
# configs are already authorized + cached.
unset HF_HOME

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/beam_equality"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/beam_equality_1b8b.log"
METHODS="bs_kernel bsk_2l_1p bsk_2l_dt"

{
    echo "=== exp1c beam-equality 1B+8B re-run started $(date -Iseconds) ==="
    echo
} | tee "$LOG"

for model_tag in Llama-3.2-1B Llama-3.1-8B; do
    echo "############ ${model_tag} ############" | tee -a "$LOG"
    export BE_MODEL="meta-llama/${model_tag}"
    export BE_DTYPE=fp16
    export BE_DUMP_BEAMS_DIR="$OUT_DIR/${model_tag}"
    mkdir -p "$BE_DUMP_BEAMS_DIR"
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

echo "Done." | tee -a "$LOG"

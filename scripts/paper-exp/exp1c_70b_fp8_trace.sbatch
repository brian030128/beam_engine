#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=dev
#SBATCH --job-name=pe_exp1c_70b_trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/paper_exp1c_70b_trace_%j.out
#SBATCH --error=slurm/logs/paper_exp1c_70b_trace_%j.err

# Re-run bs_kernel only with BS_KERNEL_TRACE_PLAN=1 on the same
# exp1c_70b_fp8 cells to capture the per-cell strategy histogram
# (which the prior exp1c run did not print). The forced variants by
# construction pin one strategy and don't need a trace.

set -uo pipefail
cd /home/u4320956/beam_engine-v3
source /etc/profile.d/lmod.sh
module load cuda/12.4
source .venv/bin/activate

export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf-${SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BE_MODEL="RedHatAI/Meta-Llama-3-70B-Instruct-FP8"
export BE_DTYPE=bf16
export BS_KERNEL_TRACE_PLAN=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/forced_2l_70b_fp8"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/exp1c_70b_fp8_trace.log"

{
    echo "=== exp1c TP=2 70B-FP8 picker-trace started $(date -Iseconds) ==="
    echo "model:    $BE_MODEL"
    echo
} | tee "$LOG"

run_cell () {
    local scenario="$1"; shift
    local B="$1"; shift
    echo "########## scenario: $scenario (B=$B, K=64) ##########" | tee -a "$LOG"
    torchrun --standalone --nproc_per_node=2 \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods bs_kernel \
        --paper-exact \
        --max-new 256 --repeat 1 --warmup \
        --scenarios "$scenario" --k-override 64 --b-override "$B" \
        --out "$OUT_DIR/trace_${scenario}.csv" \
        2>&1 | tee -a "$LOG" \
        || echo "  (cell $scenario failed; continuing)" | tee -a "$LOG"
    echo | tee -a "$LOG"
}

run_cell multi_level_system    4
run_cell multi_chain_reasoning 4
run_cell multi_few_shot        4

echo "Done. Log: $LOG" | tee -a "$LOG"

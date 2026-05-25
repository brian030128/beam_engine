#!/bin/bash
#SBATCH --account=MST114554
#SBATCH --partition=normal
#SBATCH --job-name=pe_f1234_8b_4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H100:4
#SBATCH --time=05:00:00
#SBATCH --output=slurm/logs/f1234_draws_8b_%j.out
#SBATCH --error=slurm/logs/f1234_draws_8b_%j.err

# Rerun ALL 8B final-paper cells (F1-F4) with a 3-data-draw error bar, using
# 4 GPUs to run the 4 scenarios IN PARALLEL (8B is TP=1, so one scenario per
# GPU). Each scenario does 3 data draws (seeds 0,1,2): distinct GSM8K/HotpotQA
# resamples at the SAME workload shape, --warmup discards the first iter.
#   GPU0: F1 multi_doc_qa    K=128 B=1  mn=256
#   GPU1: F2 multi_few_shot  K=128 B=4  mn=256
#   GPU2: F3 beam_search     K=64  B=8  L_p=16000 mn=256   (loops seeds)
#   GPU3: F4 self_consistency K=16 B=1  L_p=16384 mn=12000 (long pole, ~3h)
# Fresh *_draws012 CSVs; the existing n=1/n=2/n=3 files are left untouched.
# init_tp short-circuits at WORLD_SIZE=1, so each scenario is a plain python
# process pinned with CUDA_VISIBLE_DEVICES (no torchrun rendezvous to collide).
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
export BE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export BE_DTYPE=bf16          # 8B head_dim=128: FA3 compiles, no FA2 force.
export BE_KV_DTYPE=bf16

MODEL_TAG="Llama-3.1-8B-bf16"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/final_paper/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
SEEDS="0,1,2"
SGLANG_METHODS="paged fasttree deft mlca bs_kernel bsk_2l_1p bsk_2l_dt"
BATCHED_METHODS="paged fasttree deft mlca bs_kernel bs_kernel_2l1p bs_kernel_2ldt"
F4_METHODS="paged fasttree deft mlca bs_kernel bsk_2l_1p_single bsk_2l_dt"

# Pre-calibrate ONCE (GPU0) so all 4 scenarios read the same cached coeffs
# (~/.cache/beam_engine/coeffs-<gpu>-<model>-tp1.json) — avoids a concurrent
# calibrate write-race when the picker first runs in each process.
echo "=== recalibrating coefficients on GPU0 $(date -Iseconds) ==="
CUDA_VISIBLE_DEVICES=0 uv run python -m beam_engine.methods.bs_kernel.calibrate --force \
    2>&1 | tee "$OUT_DIR/run_draws_calib.log"

echo "=== launching 4 scenarios in parallel $(date -Iseconds) ==="

# --- GPU0: F1 multi_doc_qa --------------------------------------------------
(
    export CUDA_VISIBLE_DEVICES=0
    export BE_HOTPOTQA_MULTI_Q_PATH="data/hotpotqa_multi_q_K128_80k_n4.jsonl"
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $SGLANG_METHODS --paper-exact --no-stage2 \
        --max-new 256 --repeat 1 --warmup --data-seeds "$SEEDS" \
        --scenarios multi_doc_qa --k-override 128 --b-override 1 \
        --out "$OUT_DIR/F1_multi_doc_qa_K128_B1_mn256_draws012.csv"
) > "$OUT_DIR/run_draws_F1.log" 2>&1 &
PID_F1=$!

# --- GPU1: F2 multi_few_shot ------------------------------------------------
(
    export CUDA_VISIBLE_DEVICES=1
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $SGLANG_METHODS --paper-exact --no-stage2 \
        --max-new 256 --repeat 1 --warmup --data-seeds "$SEEDS" \
        --scenarios multi_few_shot --k-override 128 --b-override 4 \
        --out "$OUT_DIR/F2_multi_few_shot_K128_B4_mn256_draws012.csv"
) > "$OUT_DIR/run_draws_F2.log" 2>&1 &
PID_F2=$!

# --- GPU2: F3 beam_search (batched; one invocation per data-seed) -----------
(
    export CUDA_VISIBLE_DEVICES=2
    CSV="$OUT_DIR/F3_beam_search_K64_B8_Lp16000_mn256_draws012.csv"
    rm -f "$CSV"; first=1
    for s in 0 1 2; do
        tmp="$OUT_DIR/_f3_4gpu_s${s}.csv"; rm -f "$tmp"
        uv run python benchmarks/bs_kernel/bench_batched.py \
            --K 64 --L_p 16000 --B 8 --max_new 256 \
            --prompts-file data/hotpotqa_f3_pool.jsonl --data-seed "$s" \
            --warmup --methods $BATCHED_METHODS --out "$tmp" || echo "(F3 seed $s failed)"
        if [[ -f "$tmp" ]]; then
            if [[ $first -eq 1 ]]; then cp "$tmp" "$CSV"; first=0; else tail -n +2 "$tmp" >> "$CSV"; fi
            rm -f "$tmp"
        fi
    done
) > "$OUT_DIR/run_draws_F3.log" 2>&1 &
PID_F3=$!

# --- GPU3: F4 self_consistency (long decode, mn=12000) ----------------------
(
    export CUDA_VISIBLE_DEVICES=3
    export BE_PICKER_SINGLE_LAUNCH=1
    export BE_SELF_CONSISTENCY_LP=16384
    uv run python benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $F4_METHODS --no-stage2 \
        --max-new 12000 --repeat 1 --warmup --data-seeds "$SEEDS" \
        --scenarios self_consistency --k-override 16 --b-override 1 \
        --out "$OUT_DIR/F4_self_consistency_K16_B1_Lp16384_mn12000_draws012.csv"
) > "$OUT_DIR/run_draws_F4.log" 2>&1 &
PID_F4=$!

echo "PIDs: F1=$PID_F1 F2=$PID_F2 F3=$PID_F3 F4=$PID_F4"
rc=0
wait $PID_F1 || { echo "F1 exited nonzero"; rc=1; }
wait $PID_F2 || { echo "F2 exited nonzero"; rc=1; }
wait $PID_F3 || { echo "F3 exited nonzero"; rc=1; }
wait $PID_F4 || { echo "F4 exited nonzero"; rc=1; }
echo "=== all 8B scenarios done $(date -Iseconds) (rc=$rc) ==="
echo "CSVs:"; ls -la "$OUT_DIR"/F[1234]_*_draws012.csv
exit $rc

# Sourced by the per-model self-consistency sweep wrappers. Demonstrates
# the SHARED_2L_1POOL (single-launch) win regime: a static 2-level tree
# (one shared prefix -> K independent sampled tails, B=1), swept over
# K x L_p. The picker runs with single_launch enabled (BE_PICKER_SINGLE_LAUNCH)
# so it can realize the 1POOL win when it selects that strategy.
#
# Wrapper must define: MODEL_TAG, NPROC, METHODS (and export BE_MODEL etc).

export BE_PICKER_SINGLE_LAUNCH=1

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/self_consistency/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

{
    echo "=== self_consistency sweep ${MODEL_TAG} started $(date -Iseconds) ==="
    echo "model: $BE_MODEL (nproc=$NPROC, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo "methods: $METHODS"
    echo "picker single_launch: $BE_PICKER_SINGLE_LAUNCH"
} | tee -a "$LOG"

echo "=== recalibrating coefficients ($BE_KV_DTYPE) ===" | tee -a "$LOG"
uv run python -m beam_engine.methods.bs_kernel.calibrate --force 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"

run_cell () {
    local Lp="$1" K="$2"
    local csv="$OUT_DIR/SC_Lp${Lp}_K${K}_B1_mn256.csv"
    {
        echo
        echo "=== self_consistency  L_p=${Lp}  K=${K}  B=1 ==="
    } | tee -a "$LOG"
    export BE_SELF_CONSISTENCY_LP="$Lp"
    rm -f "$csv"
    torchrun --standalone --nproc_per_node="$NPROC" \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $METHODS \
        --no-stage2 \
        --max-new 256 --repeat 3 --warmup \
        --scenarios self_consistency --k-override "$K" --b-override 1 \
        --out "$csv" 2>&1 | tee -a "$LOG" \
        || echo "  (L_p=$Lp K=$K failed; continuing)" | tee -a "$LOG"
}

for Lp in 4096 8192 16384; do
    for K in 8 16 32; do
        run_cell "$Lp" "$K"
    done
done

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

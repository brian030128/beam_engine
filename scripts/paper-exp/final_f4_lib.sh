# Standalone runner for the F4 "best-of-N / self_consistency" cell of the
# final_paper table — the regime where SHARED_2L_1POOL (single-launch)
# wins over both DEC_TAIL and paged, and the picker selects it.
#
# Static 2-level tree: shared prefix L_p=16K -> K=16 independent sampled
# tails, B=1. Picker runs with single_launch enabled. Writes into the
# same final_paper/<MODEL_TAG>/ dir as F1-F3 so it joins the table.
#
# Wrapper defines: BE_MODEL etc (env), MODEL_TAG, NPROC,
#   SC_CORE  (gated methods; B=1 fits everywhere so no ladder),
#   SC_EXTRA (best-effort, isolated pass — fp8 baselines on 70B).

export BE_PICKER_SINGLE_LAUNCH=1

# K and L_p are env-overridable (F4_K / F4_LP) so the 70B 1p-vs-paged
# crossover can be probed at larger K without editing the lib.
K=${F4_K:-16}; B=1; MN=256; LP=${F4_LP:-16384}
export BE_SELF_CONSISTENCY_LP="$LP"
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/final_paper/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run_f4.log"
csv="$OUT_DIR/F4_self_consistency_K${K}_B${B}_Lp${LP}_mn${MN}.csv"
SC_CORE_CSV="$(echo "$SC_CORE" | tr ' ' ',')"

{
    echo "=== final_paper F4 (self_consistency) ${MODEL_TAG} started $(date -Iseconds) ==="
    echo "model: $BE_MODEL (nproc=$NPROC, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo "cell: self_consistency K=${K} B=${B} L_p=${LP} mn=${MN}, picker single_launch on"
    echo "core: $SC_CORE   extra: ${SC_EXTRA:-<none>}"
} | tee -a "$LOG"

echo "=== recalibrating coefficients ($BE_KV_DTYPE) ===" | tee -a "$LOG"
uv run python -m beam_engine.methods.bs_kernel.calibrate --force 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"

rm -f "$csv"
# CORE pass (gated single process; B=1 fits, so this is just a sanity gate).
torchrun --standalone --nproc_per_node="$NPROC" \
    benchmarks/bs_kernel/bench_sglang_e2e.py \
    --methods $SC_CORE \
    --no-stage2 --max-new "$MN" --repeat 3 --warmup \
    --scenarios self_consistency --k-override "$K" --b-override "$B" \
    --out "$csv" 2>&1 | tee -a "$LOG" || true

if uv run python scripts/paper-exp/cell_complete.py "$csv" sglang "$SC_CORE_CSV"; then
    echo "  -> F4 CORE COMPLETE" | tee -a "$LOG"
else
    echo "  -> F4 core INCOMPLETE (check log)" | tee -a "$LOG"
fi

# EXTRA best-effort pass in an isolated process (fp8 baselines on 70B).
if [[ -n "${SC_EXTRA// }" ]]; then
    etmp="${csv%.csv}.extra.csv"
    rm -f "$etmp"
    echo "  --- extra methods: $SC_EXTRA ---" | tee -a "$LOG"
    torchrun --standalone --nproc_per_node="$NPROC" \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $SC_EXTRA \
        --no-stage2 --max-new "$MN" --repeat 3 --warmup \
        --scenarios self_consistency --k-override "$K" --b-override "$B" \
        --out "$etmp" 2>&1 | tee -a "$LOG" \
        || echo "  (extra pass crashed/partial; core kept)" | tee -a "$LOG"
    [[ -f "$etmp" ]] && tail -n +2 "$etmp" >> "$csv" && rm -f "$etmp"
fi

{
    echo
    echo "=== F4 done $(date -Iseconds) ==="
    echo "Result: $csv"
} | tee -a "$LOG"

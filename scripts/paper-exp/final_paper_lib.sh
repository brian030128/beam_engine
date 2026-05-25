# Shared driver for the final_paper E2E cells. Sourced by the per-model
# sbatch wrappers, which must export/define beforehand:
#   BE_MODEL, BE_DTYPE, BE_KV_DTYPE  (env)
#   MODEL_TAG          result subdir name
#   NPROC              torchrun --nproc_per_node (1 for TP=1, 2 for TP=2)
#   SGLANG_CORE        gated sglang methods (must all fit; pick B from these)
#   SGLANG_EXTRA       best-effort sglang methods (recorded if they fit;
#                      run in a SEPARATE process so a crash can't kill core)
#   BATCHED_CORE       gated batched methods
#   BATCHED_EXTRA      best-effort batched methods
#
# Each cell walks a (K,B) ladder largest-first and accepts the first
# config where every CORE method produced a finite decode time
# (cell_complete.py). EXTRA methods are then run at that same (K,B) and
# merged in best-effort. The accepted config is encoded in the CSV name.

OUT_DIR="benchmarks/bs_kernel/results/paper-exp/final_paper/${MODEL_TAG}"
mkdir -p "$OUT_DIR" slurm/logs
LOG="$OUT_DIR/run.log"

# --- data-draw seeds ------------------------------------------------------
# Each seed resamples the underlying GSM8K/HotpotQA text for the SAME
# workload shape (the harness conforms every draw to seed-0's per-group
# token lengths; F3/F4 are shape-pinned by L_p truncation). Default "0" =
# single draw, which reproduces the original final_paper_results numbers.
SEEDS="${SEEDS:-0}"
N_DRAWS=$(echo "$SEEDS" | tr ',' '\n' | grep -c .)
MAX_SEED=$(echo "$SEEDS" | tr ',' '\n' | sort -n | tail -1)
# Per-draw timed repeats. With multiple draws the draws themselves provide
# the statistical replication, so the default drops to 1 when N_DRAWS>1.
if [[ "$N_DRAWS" -gt 1 ]]; then E2E_REPEAT="${E2E_REPEAT:-1}"; else E2E_REPEAT="${E2E_REPEAT:-3}"; fi

{
    echo "=== final_paper ${MODEL_TAG} started $(date -Iseconds) ==="
    echo "model: $BE_MODEL  (nproc=$NPROC, dtype=$BE_DTYPE, kv=$BE_KV_DTYPE)"
    echo "data draws: SEEDS=[$SEEDS] (N=$N_DRAWS), per-draw repeats=$E2E_REPEAT"
} | tee -a "$LOG"

# --- multi_doc_qa (F1) dataset: 128 questions/row, prefix capped ~80K -----
# Need >= MAX_SEED+1 distinct doc-sets so each draw reads a distinct line
# (build_multi_doc_qa picks line = b + data_seed*B). Floor of 4 matches the
# historic file. The build is deterministic (--seed 0), so the guard / any
# concurrent rebuild yields identical content.
F1_NPROMPTS=$(( MAX_SEED + 1 )); (( F1_NPROMPTS < 4 )) && F1_NPROMPTS=4
DATA_FILE="data/hotpotqa_multi_q_K128_80k_n${F1_NPROMPTS}.jsonl"
if [[ ! -s "$DATA_FILE" ]]; then
    echo "=== building $DATA_FILE (questions-per=128, cap=80000, n=$F1_NPROMPTS) ===" | tee -a "$LOG"
    uv run python scripts/data/build_hotpotqa_multi_q.py \
        --out "$DATA_FILE" \
        --questions-per 128 \
        --target-tokens 80000 \
        --cap-prefix-tokens 80000 \
        --n-prompts "$F1_NPROMPTS" \
        2>&1 | tee -a "$LOG"
fi

# --- beam_search (F3) prompt pool: one HotpotQA question per prompt -------
# Each draw uses B prompts (window usable[seed*B : seed*B+B]); B<=16, so we
# need >= N_DRAWS*16 prompts >= L_p=16000 tokens. Build at ~45k tok each so
# all clear L_p; floor 30 matches the historic file.
F3_PROMPTS="data/hotpotqa_f3_pool.jsonl"
F3_NEED=$(( N_DRAWS * 16 )); (( F3_NEED < 30 )) && F3_NEED=30
if [[ ! -s "$F3_PROMPTS" || "$(grep -c . "$F3_PROMPTS" 2>/dev/null || echo 0)" -lt "$F3_NEED" ]]; then
    echo "=== building $F3_PROMPTS (n-prompts=$F3_NEED, ~45k tok each) ===" | tee -a "$LOG"
    uv run python scripts/data/build_hotpotqa_prompts.py \
        --out "$F3_PROMPTS" --n-prompts "$F3_NEED" --target-tokens 45000 \
        2>&1 | tee -a "$LOG"
fi
# Point the multi_doc_qa builder at the K=128/85K file via env override
# (no symlink — concurrent per-model jobs share data/ and would race on a
# shared symlink). build_multi_doc_qa reads BE_HOTPOTQA_MULTI_Q_PATH.
export BE_HOTPOTQA_MULTI_Q_PATH="$DATA_FILE"
cleanup() {
    [[ -n "${HF_HOME_EPHEMERAL:-}" ]] && rm -rf "$HF_HOME"
}
trap cleanup EXIT

echo "=== recalibrating coefficients ($BE_KV_DTYPE) ===" | tee -a "$LOG"
uv run python -m beam_engine.methods.bs_kernel.calibrate --force 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"

# csv-friendly comma lists for the completeness gate (CORE only)
SGLANG_CORE_CSV="$(echo "$SGLANG_CORE" | tr ' ' ',')"
BATCHED_CORE_CSV="$(echo "$BATCHED_CORE" | tr ' ' ',')"

# --- one sglang (K,B) attempt; returns 0 only if every CORE method fit --
run_sglang_attempt () {
    local tag="$1" scenario="$2" K="$3" B="$4" mn="$5"
    local csv="$OUT_DIR/${tag}_${scenario}_K${K}_B${B}_mn${mn}.csv"
    {
        echo
        echo "=== ${tag}: ${scenario} (K=${K}, B=${B}, mn=${mn}, R=$((B*K))) ==="
    } | tee -a "$LOG"
    rm -f "$csv"
    torchrun --standalone --nproc_per_node="$NPROC" \
        benchmarks/bs_kernel/bench_sglang_e2e.py \
        --methods $SGLANG_CORE \
        --paper-exact --no-stage2 \
        --max-new "$mn" --repeat "$E2E_REPEAT" --warmup --data-seeds "$SEEDS" \
        --scenarios "$scenario" --k-override "$K" --b-override "$B" \
        --out "$csv" 2>&1 | tee -a "$LOG" || true
    if ! uv run python scripts/paper-exp/cell_complete.py "$csv" sglang "$SGLANG_CORE_CSV"; then
        echo "  -> ${tag} core incomplete at K=${K} B=${B} (shrinking)" | tee -a "$LOG"
        return 1
    fi
    echo "  -> ${tag} CORE COMPLETE at K=${K} B=${B}" | tee -a "$LOG"
    # Best-effort EXTRA methods in a separate process (isolate crashes).
    if [[ -n "${SGLANG_EXTRA// }" ]]; then
        local etmp="${csv%.csv}.extra.csv"
        rm -f "$etmp"
        echo "  --- extra sglang methods: $SGLANG_EXTRA ---" | tee -a "$LOG"
        torchrun --standalone --nproc_per_node="$NPROC" \
            benchmarks/bs_kernel/bench_sglang_e2e.py \
            --methods $SGLANG_EXTRA \
            --paper-exact --no-stage2 \
            --max-new "$mn" --repeat "$E2E_REPEAT" --warmup --data-seeds "$SEEDS" \
            --scenarios "$scenario" --k-override "$K" --b-override "$B" \
            --out "$etmp" 2>&1 | tee -a "$LOG" || \
            echo "  (extra sglang pass crashed/partial; core kept)" | tee -a "$LOG"
        [[ -f "$etmp" ]] && tail -n +2 "$etmp" >> "$csv" && rm -f "$etmp"
    fi
    return 0
}

# --- one batched (K,B) attempt; each method is its own process ----------
run_batched_attempt () {
    local tag="$1" K="$2" B="$3" Lp="$4" mn="$5"
    local csv="$OUT_DIR/${tag}_beam_search_K${K}_B${B}_Lp${Lp}_mn${mn}.csv"
    {
        echo
        echo "=== ${tag}: beam_search (K=${K}, B=${B}, L_p=${Lp}, mn=${mn}) ==="
    } | tee -a "$LOG"
    rm -f "$csv"
    local first=1
    # One torchrun process per (data-draw seed, method). The per-seed prompt
    # window (usable[seed*B : seed*B+B]) gives each draw distinct HotpotQA
    # prompts; L_p truncation pins the shape across draws.
    for s in ${SEEDS//,/ }; do
    for m in $BATCHED_CORE $BATCHED_EXTRA; do
        echo "  --- method: $m (seed=$s) ---" | tee -a "$LOG"
        local tmp="$OUT_DIR/${tag}_${m}_K${K}_B${B}_s${s}.tmp.csv"
        rm -f "$tmp"
        torchrun --standalone --nproc_per_node="$NPROC" \
            benchmarks/bs_kernel/bench_batched.py \
            --K "$K" --L_p "$Lp" --B "$B" --max_new "$mn" \
            --prompts-file "$F3_PROMPTS" --data-seed "$s" \
            --warmup --methods "$m" --out "$tmp" 2>&1 | tee -a "$LOG" || true
        if [[ -f "$tmp" ]]; then
            if [[ $first -eq 1 ]]; then cp "$tmp" "$csv"; first=0
            else tail -n +2 "$tmp" >> "$csv"; fi
            rm -f "$tmp"
        fi
    done
    done
    # Gate on CORE only; EXTRA methods are best-effort.
    if uv run python scripts/paper-exp/cell_complete.py "$csv" batched "$BATCHED_CORE_CSV"; then
        echo "  -> ${tag} CORE COMPLETE at K=${K} B=${B}" | tee -a "$LOG"
        return 0
    fi
    echo "  -> ${tag} core incomplete at K=${K} B=${B} (shrinking)" | tee -a "$LOG"
    return 1
}

ladder_sglang () {  # tag scenario mn  "K B" "K B" ...
    local tag="$1" scenario="$2" mn="$3"; shift 3
    for kb in "$@"; do
        set -- $kb
        if run_sglang_attempt "$tag" "$scenario" "$1" "$2" "$mn"; then return 0; fi
    done
    echo "  !! ${tag} could not fit any (K,B)" | tee -a "$LOG"
}

ladder_batched () {  # tag Lp mn  "K B" "K B" ...
    local tag="$1" Lp="$2" mn="$3"; shift 3
    for kb in "$@"; do
        set -- $kb
        if run_batched_attempt "$tag" "$1" "$2" "$Lp" "$mn"; then return 0; fi
    done
    echo "  !! ${tag} could not fit any (K,B)" | tee -a "$LOG"
}

# F1 multi_doc_qa: B=1, just drop K if the long prefix won't fit.
ladder_sglang  F1 multi_doc_qa   256  "128 1" "64 1"
# F2 multi_few_shot: K=128 with lower B (B=64/32 OOM on every model);
# ladder from 16 down, max-fit per model. Standard paper-exact cell; with the
# FusedMultiLevelCascade CTA_TILE_Q fix the picker should land on
# SHARED_2L_DEC_TAIL (its fused prefix now tiles at CTA_TILE_Q=64).
ladder_sglang  F2 multi_few_shot 256  "128 16" "128 8" "128 4" "128 2" "128 1"
# F3 beam_search: K=64, 16K prefix (down from 40K), shrink B to fit.
ladder_batched F3 16000 256       "64 16" "64 8" "64 4" "64 2" "64 1"

{
    echo
    echo "=== done $(date -Iseconds) ==="
    echo "Results: $OUT_DIR"
} | tee -a "$LOG"

# Re-run paged-only on the F1-F4 final_paper cells with the LCA index
# dedup, merging the new paged rows into the canonical CSVs (replacing the
# pre-optimization paged rows). Flags/env per cell must match the original
# run so paged's workload is identical to the other methods in that cell.
#
# Wrapper defines: BE_MODEL etc (env), MODEL_TAG, NPROC, F2_B, F3_B
# (settled max-fit batch sizes for this model).

export BE_HOTPOTQA_MULTI_Q_PATH=data/hotpotqa_multi_q_K128_80k.jsonl
OUT_DIR="benchmarks/bs_kernel/results/paper-exp/final_paper/${MODEL_TAG}"
LOG="$OUT_DIR/run_paged_rerun.log"
mkdir -p "$OUT_DIR" slurm/logs
echo "=== paged re-run (LCA dedup) ${MODEL_TAG} $(date -Iseconds) ===" | tee -a "$LOG"

rerun_sglang () {  # tag scenario K B canon flags
    local tag="$1" scen="$2" K="$3" B="$4" canon="$5" flags="$6"
    local tmp="$OUT_DIR/_paged_${tag}.csv"
    echo "=== $tag: paged $scen K=$K B=$B [$flags] ===" | tee -a "$LOG"
    rm -f "$tmp"
    torchrun --standalone --nproc_per_node="$NPROC" \
        benchmarks/bs_kernel/bench_sglang_e2e.py --methods paged $flags \
        --max-new 256 --repeat 3 --warmup --scenarios "$scen" \
        --k-override "$K" --b-override "$B" --out "$tmp" 2>&1 | tee -a "$LOG" \
        || echo "  ($tag paged failed)" | tee -a "$LOG"
    uv run python scripts/paper-exp/merge_method.py "$OUT_DIR/$canon" "$tmp" paged 2>&1 | tee -a "$LOG"
}

# F1 multi_doc_qa, F2 multi_few_shot: paper-exact (matches original run)
rerun_sglang F1 multi_doc_qa   128 1     "F1_multi_doc_qa_K128_B1_mn256.csv"        "--paper-exact --no-stage2"
rerun_sglang F2 multi_few_shot 128 "$F2_B" "F2_multi_few_shot_K128_B${F2_B}_mn256.csv" "--paper-exact --no-stage2"

# F4 self_consistency: plain (no paper-exact), L_p=16K
export BE_SELF_CONSISTENCY_LP=16384
rerun_sglang F4 self_consistency 16 1 "F4_self_consistency_K16_B1_Lp16384_mn256.csv" "--no-stage2"
unset BE_SELF_CONSISTENCY_LP

# F3 beam_search: batched harness
echo "=== F3: paged beam_search K=64 B=${F3_B} L_p=16000 ===" | tee -a "$LOG"
tmp3="$OUT_DIR/_paged_F3.csv"; rm -f "$tmp3"
torchrun --standalone --nproc_per_node="$NPROC" \
    benchmarks/bs_kernel/bench_batched.py --K 64 --L_p 16000 --B "$F3_B" --max_new 256 \
    --prompts-file data/hotpotqa.jsonl --warmup --methods paged \
    --out "$tmp3" 2>&1 | tee -a "$LOG" || echo "  (F3 paged failed)" | tee -a "$LOG"
uv run python scripts/paper-exp/merge_method.py \
    "$OUT_DIR/F3_beam_search_K64_B${F3_B}_Lp16000_mn256.csv" "$tmp3" paged 2>&1 | tee -a "$LOG"

echo "=== paged re-run done $(date -Iseconds) ===" | tee -a "$LOG"

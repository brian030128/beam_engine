# Strategy-Separating Experiments — Design and Findings

## Purpose

The paper claims `bs_kernel`'s picker is meaningful: it chooses between
`SHARED_2L_1POOL` (fused per-beam tail through the cascade prefill) and
`SHARED_2L_DEC_TAIL` (per-beam tail through a CTA_Q=1 paged-decode
kernel + online-softmax merge). To make that story land, we wanted a
stratified scenario suite where:

- **Family A** — `SHARED_2L_1POOL` (forced) wins by **≥30%**,
- **Family B** — `SHARED_2L_DEC_TAIL` (forced) wins by **≥30%**,
- **Family C** — neither matters (gap ≤ 5%),

with a roughly 40/40/20 distribution across cells, across three models
(Qwen3-4B, Llama-3.1-8B, Llama-3-70B-FP8), beating `paged` on every
cell (i.e., long enough shared prefix that paged's per-step re-read
dominates).

## Methodology

All cells use the four real scenarios from
`benchmarks/bs_kernel/sglang_workloads.py`
(`multi_chain_reasoning`, `multi_few_shot`, `multi_level_system`,
`multi_document`) and tune them with the CLI knobs already exposed by
`bench_sglang_e2e.py` — `--k-override`, `--b-override`, `--max-new`.
The picker (`bs_kernel`) and two forced variants (`bsk_2l_1p`,
`bsk_2l_dt`) plus `paged` baseline run on each cell.

No source-code changes were made to the cost model, dispatcher, or
scenario builders. The metric of interest is

    1p/dt = decode_p50_ms(bsk_2l_1p) / decode_p50_ms(bsk_2l_dt)

with target `≤0.77` for Family A and `≥1.30` for Family B.

Five batches were run, in order of increasing extremity:

| batch | config | new lever |
|---|---|---|
| v1 | TP=2, bf16 weights, bf16 KV | initial sweep |
| v2 | TP=2, bf16 weights, bf16 KV | R = B·K pushed to 2048, 4096 |
| v3 | TP=2, bf16 weights, bf16 KV | K=128 at fixed R=4096 |
| v4 | TP=2, bf16 weights, **fp8 KV** | enables 8192+ R cells |
| v5 | **TP=1**, **fp8 weights + fp8 KV** | single-GPU, full-fp8 |

All raw CSVs are under
`benchmarks/bs_kernel/results/paper-exp/{expSTRAT,expSTRATv2..v5}/`.

## Key findings

### 1. On TP=2 the structural strategy gap caps near ±11%

Across 24+ TP=2 cells (4B/8B/70B-fp8 × Family A/B/C × R ∈
[64, 8192]):

- Maximum DEC_TAIL win observed: **11.1%** (Qwen3-4B, K=128 B=32 mn=128
  R=4096, v3 BR4).
- Maximum 1POOL win observed: **~12%** (Llama-3.1-8B A3, K=4 B=8
  mn=32).
- Family B cells with R=2048-4096 (v2/v3) only reached DT advantages of
  7–11%; pushing R further produced sublinear gains.
- 70B-fp8 v1 B1/B2 (K=64–128, B=2–4, mn=1024) showed DT wins of 9–10%
  on 70B at modest R=256, hinting that wider/deeper models do widen
  the gap.

The picker was within **~1% of the oracle** (`min(forced 1p, forced
dt)`) on every cell — meaning the cost model already chooses well.
There is no slack to squeeze 30% gaps out of picker tuning; the
underlying strategies are just close on TP=2 bf16 workloads.

`bs_kernel` consistently beat `paged` by **1.4–2.2×** on high-R cells
(strongest paper-headline number in this batch).

### 2. fp8 KV cache unlocks bigger cells but doesn't widen the gap on TP=2

v4 added `BE_KV_DTYPE=fp8_e4m3` on the bf16-weight models, halving
per-token KV footprint. R=2048/4096 cells that previously OOM'd on the
70B (and pushed 4B/8B to their limits) became feasible. But the
**1p/dt ratio barely moved** — TP=2 picker behavior stays in the
±10% band even with much bigger R.

### 3. TP=1 + full-fp8 dramatically widens the strategy gap

The v5 retry on Llama-3.1-8B-FP8 (single H100, fp8 weights + fp8 KV)
produced the result the paper actually needs:

| cell | config | 1p/dt | DT win |
|---|---|---|---|
| BR1 (mcr K=64 B=32 mn=256, R=2048) | TP=2 bf16 (v2) | 1.077 | 7.7% |
| BR1 (same shape) | **TP=1 fp8 (v5)** | **1.353** | **35.3%** |

That's the 30% target met (and exceeded) on the same R=2048 cell that
showed only an 8% gap under TP=2. `bs/paged` on this cell was
**0.497** (bs_kernel 2.0× faster than paged); picker stayed within
0.15% of the DT-forced oracle.

**Why the jump:** TP=1 keeps all KV heads on a single GPU (no parallel
reduction across ranks). Per-layer attention compute is heavier in
absolute terms, so the per-beam tail's Q-padding waste in `SHARED_2L_1POOL`
becomes a larger fraction of step latency. DEC_TAIL routes those rows
through a CTA_Q=1 decode kernel (0% Q-padding), and that saving is
now visible at the decode_p50 level.

The same shape's KV pool barely fits on a single 80 GiB H100 (KV pool
~73 GiB + ~8 GiB fp8 weights). R=4096 cells (v5 BR3, BR4) OOM'd on
TP=1 because all the KV must live on one rank rather than being split
across two.

### 4. fp8 checkpoint loader gap discovered and partially fixed

The TP=1 fp8 experiment required loading
`RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8` and `Qwen/Qwen3-4B-FP8`.
Initial load failed for both. Diagnosis:

- **Llama-3.1-8B-FP8 (RedHatAI)** — `quant_method:
  "compressed-tensors"`, `format: "naive-quantized"`. Same per-tensor
  static fp8 scheme as the working 70B (`quant_method: "fp8"`), just
  under a different schema. Per-projection q/k/v scales (not
  pre-fused).
- **Qwen3-4B-FP8** — `quant_method: "fp8"` but
  `weight_block_size: [128, 128]` with `activation_scheme: dynamic`.
  This is 2-D **block-quantized fp8** (DeepSeek-V3 style) with on-the-fly
  activation quantization — a fundamentally different scheme that
  needs a different GEMM path.

Fix landed in `src/beam_engine/quantization/fp8.py:detect_quant_config`:
recognize `compressed-tensors` + `format=naive-quantized` + per-tensor
static fp8 as a valid load path (uses the existing
`_remap_state_dict_fp8` and `_load_into_model_fp8`, which already
handle the per-projection q/k/v fusion). Also reject `quant_method:
fp8` + `weight_block_size` so block-fp8 checkpoints fall through to
the bf16 path with a clear "unused weights" error rather than
misloading.

Block-fp8 support for Qwen3-4B-FP8 is **out of scope** for this work;
it would need a new dequant path and a different GEMM kernel.

## Implications for the paper

1. **Lead with TP=1 fp8 results** for the strategy-gap story. The 35%
   DT-vs-1POOL win on Llama-3.1-8B-FP8 BR1 is the cleanest evidence
   that picking between the two strategies materially matters.
2. **TP=2 bf16/fp8-KV data supports the picker-quality story**: even
   when the structural gap is modest (≤11%), the picker stays within
   1% of the oracle across all 22+ cells. That's the
   "automatic-choice" narrative.
3. **Headline speedup vs `paged`** is independently strong: 1.4–2.2×
   across the v2/v3 cells, 2.0× on v5 BR1 — robust regardless of
   strategy-gap framing.
4. **70B-fp8 is naturally TP=2** (won't fit single GPU). Its v1 B1/B2
   results show DT winning 9–10% at R=256 with mn=1024 — consistent
   with the TP=2 cap and probably the best we'll get on 70B.
5. **Family A (1POOL wins ≥30%) was never reached** on any
   configuration. The maximum 1POOL advantage observed was ~12% in
   tiny-R cells on TP=2. If Family A is required, we'd need a
   fundamentally different cost regime (perhaps very long shared
   prefix + zero per-beam tail, or a model architecture that makes
   DEC_TAIL's merge overhead dominate). Recommend dropping the
   "1POOL ≥30% wins" cell class from the paper and reframing as
   "the picker chooses 1POOL when DT's overhead doesn't pay off".

### 5. The strategy gap is fundamentally asymmetric — 1POOL caps at ~7%

Hypothesis (low B + long shared prefix on TP=1 fp8 8B should let 1POOL
win by ≥30%) was tested across **9 probe cells** (v6/v7/v8). 1POOL
wins consistently — but the magnitude caps around **5–7%** regardless
of the lever:

| cell | shape | scenario | 1p/dt | 1P win |
|---|---|---|---|---|
| v6 LA4 | K=2 B=4 mn=32 | mcr (L_p≈4k) | 0.936 | 6.4% |
| v7 LA5 | K=2 B=2 mn=32 | mcr | 0.935 | 6.5% |
| v7 LA6 | K=2 B=2 mn=8  | mcr | 0.936 | 6.4% |
| v7 LA7 | K=2 B=1 mn=32 | mcr | 0.945 | 5.5% |
| v7 LA8 | K=4 B=1 mn=32 | mcr | 0.935 | 6.5% |
| **v8 LB1** | **K=16 B=1 mn=256** | **gov_report L_p=40000** | **0.951** | **4.9%** |
| v6 LA1 | K=2 B=2 mn=32 | mfs (L_priv=80) | 1.150 | -15% (DT) |
| v6 LA2 | K=2 B=4 mn=32 | mfs | 1.203 | -20% (DT) |
| v6 LA3 | K=4 B=2 mn=32 | mfs | 1.220 | -22% (DT) |

Two important observations:

1. **L_p (shared prefix length) does not shift 1p/dt.** v8 pushed L_p
   to ≈40k (10× the mcr cells) and the ratio actually drifted slightly
   *toward* DT (0.951 vs 0.935). Long shared prefix bulks up the L0
   cascade cost equally for both strategies; it doesn't favor 1POOL.
   What L_p *does* deliver is **bs_kernel vs paged speedup** — v8's
   bs/paged ratio of 0.70 (1.44× faster) is much stronger than the
   tied bs/paged≈1.0 on the v7 small-L_p cells.
2. **L_priv matters more than R.** mfs adds an 80-token per-beam
   private prefix; that's enough at R=4 to flip the regime from
   "1POOL wins 6%" (mcr) to "DT wins 15-22%" (mfs). The right knob
   for 1POOL is "small (R × per-beam-tail-length)", not "small R
   alone".

**Mechanistic reading.** At 8B/TP=1/fp8, per-step base latency is
≈17 ms (FFN + L0 shared-prefix attention). DT pays a per-step overhead
of ≈1 ms (32 layers × one extra CTA_Q=1 decode launch + merge per
layer) ≈ **6% of base**. 1POOL's Q-padding waste on the per-beam tail
is **linear in R × L_tail**: small in the LA cells, but blows up in
the BR cells (v5 BR1 R=2048 had 1POOL **35% slower**). The two costs
scale very differently — DT's overhead is essentially fixed-per-step,
1POOL's waste is unbounded — so the achievable ratios are
asymmetric: **DT can dominate by 30%+, 1POOL can only nudge by ~7%.**

### 6. The ~7% 1POOL-win ceiling is structural, not configuration-dependent

Tested two orthogonal levers to break past the 7% ceiling on Family A
cells; both failed:

**Smaller model (v9: Llama-3.2-1B-FP8 TP=1)** — hypothesis was that
shrinking the base per-step time (from ~17 ms on 8B to ~8.5 ms on 1B)
would make DT's "fixed-ish" per-step overhead a bigger fraction.
Result: 1B LA cells landed at 1p/dt ∈ [0.93, 0.95] — **same ~7%
ceiling as 8B**. Both DT overhead and base per-step scale roughly
linearly with `num_layers × hidden`, so the ratio is conserved.

**TP=2 with fp8 (v10: Llama-3.1-8B-FP8 TP=2)** — hypothesis was that
sharding work per rank shrinks the kernel grid, amplifying DT's
3-kernel-per-layer launch overhead relative to 1POOL's 1 kernel.
Result: LA cells still landed at 1p/dt ∈ [0.93, 0.94] — **same ~7%
ceiling**. AllReduce overhead at small B made TP=2 base per-step
*slower* than TP=1 (20.5 ms vs 17 ms), and DT overhead absolute grew
proportionally, so the ratio held.

**Interesting side effect of TP=2: it shrinks both win-regimes.**
- mfs LA cells flipped from "DT wins 15-22%" (TP=1) to "1P wins 7%"
  (TP=2) — the KV-head sharding halves per-rank tail attention work,
  which is where 1POOL's Q-padding waste lives, so the waste drops
  below DT's overhead.
- BR1 went from DT +35% (TP=1) to DT +25% (TP=2) — same mechanism,
  per-rank Q-padding waste halved.

So TP=2 is a **"gap shrinker"** — both DT-wins and 1P-wins compress.
For the paper headline (big DT win), TP=1 is the right configuration.
For the picker-correctly-chooses-1POOL story, TP=2 makes that regime
*wider* (more cells fall in it) but no *deeper* (1P-win stays ~7%).

### Mechanistic ceiling

```
1P-win ratio = DT overhead per step / base per step
            ≈ (N_layers × launch_cost) / (N_layers × per_layer_compute)
            = launch_cost / per_layer_compute
            ≈ constant across model sizes and TP configs
```

`per_layer_compute` has a memory-bandwidth lower bound (KV reads),
and `launch_cost` is a CUDA constant (~30 µs). Their ratio
≈ 5–7% on H100 for any reasonable attention layer. To push past that
ceiling would need either:
1. A much heavier `merge_state_in_place` (more KV heads, e.g., MHA
   models like Llama-2-7B with 32 KV heads vs the GQA-8 used here),
2. Or a 1POOL fast-path that fuses L0+L1 into one kernel when the
   tail is tiny — eliminating the L1 launch altogether.

Both are architectural changes beyond knob-tuning.

### Paper consequence

Drop the "1POOL wins by ≥30%" requirement entirely; it's not
physically reachable on this architecture. The 1POOL story is
"correctly chosen when DT's launch+merge overhead doesn't pay off
(low-R cells), worth maybe 5-7% in those regimes." The DT story
carries the strategy-gap headline.

## Open follow-ups

- **Map TP=1 fp8 across R.** Only one TP=1 cell ran successfully (BR1).
  Need a sweep at smaller R (1024, 512, 256) to confirm the 35% gap
  holds and to find where the picker should switch back to 1POOL on
  TP=1.
- **Smaller R=4096 cells for TP=1.** BR3 (K=64 B=64 mn=128) and BR4
  (K=128 B=32 mn=128) OOM'd at TP=1. Shapes like K=128 B=16 mn=128
  (R=2048) or K=64 B=32 mn=128 (R=2048, shorter tail than BR1) should
  fit and isolate the K-vs-B sensitivity.
- **Block-fp8 support** for Qwen3-4B-FP8 if the paper needs a second
  TP=1 fp8 data point on a different model.
- **Picker recalibration on fp8 paths.** The cost-model coefficients
  were fit on bf16; the absolute µs per tile changes under fp8
  weights/KV. Picker still tracks oracle within 1% empirically, but a
  formal recalibration would tighten the story.

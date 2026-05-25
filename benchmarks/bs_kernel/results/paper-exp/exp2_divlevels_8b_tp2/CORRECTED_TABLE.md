# Dispatch-space depth ablation — corrected (Llama-3.1-8B, TP=2)

Cell: gov_report long-decode beam search, K=32, L_p=8192, B=8, max_new=2048.
Forward time measured over decode steps 1500–2000. Source job: 212757.

## Why the original (`exp2_tp2_8b`) numbers were wrong

The deeper force-aliases listed shallower strategies in `available_strategies`,
so the cost model picked the cheapest one. The original run's own
`BS_KERNEL_TRACE_PLAN` showed `4ldt`/`5ldt` ran **100 % SHARED_2L_DEC_TAIL**
(and `4l1p` reached 4L for only 166/2047 steps): the deep DECTAIL rows were
2L measured again with run-to-run noise, not genuine 4L/5L.

## What changed

1. Deeper cascade levels now come from **beam-search divergence**: a fork
   becomes its own shared cascade level (`_adaptive_levels`). The level-creation
   gate (`BS_KERNEL_MIN_LEVEL_CHILDREN`/`_TOKENS`) is set to its natural minimum
   (≥2 beams sharing ≥1 page). gov_report beam divergence is fine-grained —
   built from 2-beam clusters over 1–2 page runs — so any stricter gate
   (e.g. >8 children / >128 tokens, or even >2 children / >32 tokens) suppresses
   100 % of the deep structure and collapses every depth to 2L. The natural
   minimum is the only setting that lets each forced depth run at that depth.
2. Each forced depth is **hard-forced** to its single strategy family; the
   cost model auto-falls-back to a shallower depth only when the beam tree
   genuinely lacks that many levels.

## Realized depth distribution (of 2047 decode steps)

| forced | depth actually run |
|---|---|
| 3L_FUSED   | 3L 2020 (99 %) |
| 4L_FUSED   | 4L 1892 (92 %) |
| 5L_FUSED   | 5L 1683 (82 %) |
| 3L_DECTAIL | 3L 2024 (99 %) |
| 4L_DECTAIL | 4L 1850 (90 %) |
| 5L_DECTAIL | 5L 1730 (85 %) |

## Corrected table

| Strategy   | Fwd. time (ms) | Imp. vs 2L |
|------------|---------------:|-----------:|
| 2L_FUSED   | 12,372 | — |
| 3L_FUSED   | 12,405 | +0.3 % |
| 4L_FUSED   | 12,323 | −0.4 % |
| 5L_FUSED   | 11,702 | −5.4 % |
| 2L_DECTAIL | 10,739 | — |
| 3L_DECTAIL | 11,651 | +8.5 % |
| 4L_DECTAIL | 10,731 | −0.1 % |
| 5L_DECTAIL | 10,999 | +2.4 % |

(Picker `bs_kernel` chose SHARED_2L_DEC_TAIL for 100 % of steps → 10,840 ms.)

## Notes

- Absolute forward times sit above the May-19 `exp2_tp2_8b` run (e.g. 2L_DECTAIL
  10,739 vs 8,813) because the DEC_TAIL / cascade kernels changed since then
  (CTA_TILE_Q fix and related work, 2026-05-24). The ablation is the *within-run*
  per-depth comparison, which is now genuine.
- The 3L_DECTAIL penalty cross-validates: +8.5 % here vs +6.9 % in the original
  (the one deep row that *was* genuine in the buggy run).

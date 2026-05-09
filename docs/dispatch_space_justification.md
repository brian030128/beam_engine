# Dispatch-space justification (post cost-model fix)

## TL;DR

The "11 candidates" framing in `paper_story.md` happens to have the
right cardinality but the wrong composition. Replace:

- ❌ Production-11: `PER_BEAM ∪ {SHARED_{2,3}L_{1,2}POOL × T_large ∈ {64,128}} ∪ SHARED_{2,3}L_DEC_TAIL`
- ✅ Empirical-11: `PER_BEAM ∪ {SHARED_dN_p1_T16 : N ∈ 2..6} ∪ {SHARED_dN_DEC_TAIL : N ∈ 2..6}`

Three structural changes:
1. **Drop 2-pool variants**: dominated by 1-pool-T16 or DEC_TAIL on every workload tested.
2. **T_large = 16 only** for 1POOL: T={64, 128} are dominated under the calibrated cost model.
3. **Extend depth from {2, 3} to {2..6}**: depth>3 wins on 41.9% of synthetic hierarchical workloads.

## What changed: the BW-efficiency fix

`Coefficients.bw_efficiency_floor` (default 1.0 = legacy) penalizes
prefill tiles at low utilization (per-beam-tail-like situations
where `packed=1` and `T=16`). Decode kernel (DEC_TAIL's tail) bypasses
this — it stays at peak BW.

H100-calibrated value: `floor=0.5`. Validation against 16 measured
cells: match rate 19% → 81%, mean regret 7.85% → 0.41%, max regret
38% → 2.83%.

## Empirical study: which candidates does the picker actually choose?

After the fix, on a 4752-cell synthetic grid (K∈{16,32,64} ×
L_p∈{1K..64K} × n_intermediate∈{0..4} × tail_len∈{16,64,256,1024}
× B∈{1,8,32}, candidate space = 56 strategies including
T∈{16,32,64,96,128} × pool∈{1,2} × depth∈{2..6}):

| Family | % of cells | Strategy | Per-strategy %
|---|---|---|---|
| **DEC_TAIL** | **70.5%** | shared_d2_dec_tail | 23.7% |
| | | shared_d3_dec_tail | 21.6% |
| | | shared_d4_dec_tail | 14.4% |
| | | shared_d5_dec_tail | 7.7% |
| | | shared_d6_dec_tail | 3.0% |
| **SHARED_*L_1POOL_T16** | **27.2%** | shared_d4_p1_t16 | 6.8% |
| | | shared_d3_p1_t16 | 6.4% |
| | | shared_d5_p1_t16 | 5.6% |
| | | shared_d6_p1_t16 | 4.4% |
| | | shared_d2_p1_t16 | 4.0% |
| **PER_BEAM** | **2.3%** | per_beam | 2.3% |

**Exactly 11 distinct strategies are ever picked** across 4752 cells.
Cardinality matches the paper's "11" claim — but composition differs:

| Axis | Production | Empirical |
|---|---|---|
| Pool count | {1, 2} (4 candidates × ...) | {1} only |
| T_large | {64, 128} (× 2 candidates) | {16} only |
| Depth (SHARED) | {2, 3} | {2, 3, 4, 5, 6} |
| DEC_TAIL depths | {2, 3} | {2, 3, 4, 5, 6} |

## Why 2-pool is dominated

| dual_pool_extra_us | 2-pool win% (no DEC_TAIL) | 2-pool win% (full space) |
|---|---|---|
| 0 (no penalty) | **20.5%** | 1.9% |
| 5 | 0.0% | 0.0% |
| 10 | 0.0% | 0.0% |
| 20 (calibrated H100) | 0.0% | 0.0% |
| 50 | 0.0% | 0.0% |

At calibrated `dual_pool_extra_us = 20µs`, 2-pool never wins. Even
at zero overhead, 2-pool wins on at most 20% of cells *without
DEC_TAIL* — DEC_TAIL still wins in that regime.

**Real-measurement note**: at K=64/L_p=32K/B=8 we measured 2L_2POOL
= 2.77 ms vs 2L_1POOL = 2.96 ms (2-pool wins by 6%). But DEC_TAIL =
2.34 ms beats both. So the 2-pool win is real but irrelevant — DEC_TAIL
captures that regime regardless. **Conclusion: drop 2-pool variants.**

## Why T_large = 16 dominates for 1POOL

Under the fixed cost model, the per-beam tail level (packed=1) at
T=16 has util=6.25% and the BW penalty applies. At T=64 or T=128,
util is even lower (1.5%, 0.8%) but bytes loaded are the same.
The model's compute term `waves × per_tile_us[T]` makes T=16 cheaper
in the high-tile-count waves regime.

In the cost model's view, T=16 dominates T=64 for 1POOL because:
- Bytes loaded: identical (level KV is fixed)
- Per-tile compute: T=16 (0.6 µs) << T=128 (2.4 µs)
- Waves: similar (both saturate SMs at high B×K)

**Real-measurement note**: production picker today only enumerates
T∈{64, 128}. We haven't measured T=16 single-pool on real hardware
(would require extending production picker). But the fixed cost
model's preference for T=16 single-pool over T=64 single-pool on
the per-beam-tail-dominated workloads matches the structural intuition
(less per-tile padding waste).

## Why depth > 3 wins on 42% of cells

Cost-model prediction only — not yet validated by real measurements
(would require extending FlashInfer wrapper for `num_levels > 3`).

Depth distribution on the synthetic grid:

| Depth | Cells (%) |
|---|---|
| 1 (per_beam) | 2.3% |
| 2 | 27.7% |
| 3 | 28.1% |
| **4** | **21.2%** |
| **5** | **13.3%** |
| **6** | **7.4%** |

Depth>3 wins specifically on workloads with hierarchical sharing
(non-zero intermediate levels). For workloads where the LCA cap is 2
(no intermediate sharing), depth=2 wins; with one intermediate
level, depth=3 wins; with N-1 intermediates available, depth=N can
win. The cost model amortizes the extra plan-launch + merge cost
of deeper cascades against the bandwidth saved at lower-cardinality
groups.

**This is the most important unvalidated claim.** Real-measurement
gating is needed before committing to depth>3 in the paper.

## Recommended dispatch space for the paper

**Conservative (production-supported today)**:
```
{PER_BEAM} ∪ {SHARED_dN_p1_T16 : N ∈ {2, 3}} ∪ {SHARED_dN_DEC_TAIL : N ∈ {2, 3}}
```
5 candidates. Validated by H100 measurements at the cells we tested.
Drop 2-pool, drop T={64, 128} 1POOL.

**Empirically-suggested (post fix, depth-extended)**:
```
{PER_BEAM} ∪ {SHARED_dN_p1_T16 : N ∈ 2..6} ∪ {SHARED_dN_DEC_TAIL : N ∈ 2..6}
```
11 candidates. Coincidentally matches the paper's "11" claim.
**Depth>3 unvalidated** — gates on extending the FlashInfer wrapper.

**Generalized (per workload's depth_max)**:
```
{PER_BEAM} ∪ {SHARED_dN_p1_T16 : N ∈ 2..D_max} ∪ {SHARED_dN_DEC_TAIL : N ∈ 2..D_max}
```
1 + 2 × (D_max - 1) candidates. D_max determined by the workload's
LCA structure cardinality.

## What to write in the paper

- The dispatch space has **3 axes**, not 4: `(family, depth)`
  where family ∈ `{PER_BEAM, SHARED_*L_1POOL_T16, SHARED_*L_DEC_TAIL}`
  and depth ∈ `{2..D_max}`. (T per pool collapses to T=16 only;
  pool count collapses to 1; both axes discovered empirically.)
- The cardinality is `1 + 2(D_max - 1)`, not 11. For typical
  branch-tree workloads (D_max=3), it's 5; for hierarchical
  workloads with deeper sharing (D_max=6), it's 11.
- 2-pool and T_large∈{32,64,96,128} are *dominated* — explicitly
  show the cost-model and measurement evidence and exclude from
  the space.
- Depth>3 is gated on real-measurement validation (extend FlashInfer
  wrapper to test `num_levels > 3`). Without that validation, the
  paper should claim the conservative space (D_max=3, 5 candidates).

## Caveats

1. **Depth>3 is unvalidated by real measurements.** Cost-model
   predicts depth>3 wins on hierarchical workloads, but no kernel
   measurements exist at depth>3 (FlashInfer wrapper limited to 3
   levels in the production code path). This is the next experiment.

2. **Synthetic uniform-branching workloads** are stylized. Real
   DBS/spec-decoding hierarchies aren't perfectly uniform. The
   directional findings hold; magnitudes may shift on real workloads.

3. **PER_BEAM still over-estimated** at small K and B=1: cost model
   doesn't account for L2-caching of the shared prefix across the K
   beams within a single prompt. Separate fix (cost_per_beam_batch)
   needed; not addressed by this BW-efficiency change.

4. **2-pool's dominance result is sensitive to `dual_pool_extra_us`.**
   At calibrated 20µs, 2-pool never wins. At 0µs, 2-pool wins on
   20% (no DEC_TAIL) → 1.9% (with DEC_TAIL). DEC_TAIL captures the
   2-pool regime regardless.

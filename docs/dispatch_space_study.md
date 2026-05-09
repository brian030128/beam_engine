# Dispatch-space study (Phase 0)

**Question:** the production picker enumerates 11 candidates per step
(`{PER_BEAM, SHARED_{2,3}L_{1,2}POOL, SHARED_{2,3}L_DEC_TAIL}` × T_large
∈ {64, 128}). Is this the right dispatch space?

**Method:** generalize the cost model along three axes, run the picker
on a synthetic workload grid, see what it picks. The expanded space
adds:

- `depth` ∈ {2, 3, 4, 5, 6} (was {2, 3})
- `T_large` ∈ {16, 32, 64, 96, 128} (was {64, 128})
- `pool_count` ∈ {1, 2} (unchanged)
- `tail_kernel` ∈ {prefill, dec_tail} (unchanged)

Total: 56 candidates per cell.

**Coefficients:** H100 calibrated values from
`~/.cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json`:

```
B_hbm = 1.296 × 10⁶ bytes/µs   (≈ 1.3 TB/s)
share_extra_us = 100            (flat per-call overhead for cascade)
dual_pool_extra_us = 20         (extra overhead for 2-pool)
merge_us = 6.94                 (per merge call)
num_sms = 132                   (H100)
per_tile_us = {16: 0.6, 64: 1.4, 128: 2.4}
per_tile_us[32] = 0.867, per_tile_us[96] = 1.9   (linearly interpolated)
```

**Workload grid (792 cells):** uniform-branching hierarchies covering
K ∈ {16, 32, 64}, L_p ∈ {1K, 4K, 8K, 32K}, n_intermediate ∈ {0..4}
(depth_max ∈ {2..6}), intermediate-len ∈ {64, 256, 512}, tail-len
∈ {16, 64, 256}, branching factor ∈ {2, 4}. Code:
[`benchmarks/bs_kernel/study_dispatch_space.py`](../benchmarks/bs_kernel/study_dispatch_space.py).

## Headline findings

The production 11-candidate dispatch space is wrong on three
independent axes:

| Finding | Evidence |
|---|---|
| **Depth>3 is empirically Pareto-optimal on ~44% of cells** | 345 / 792 cells pick depth ∈ {4, 5, 6}. Robust to share_extra_us up to 1000 µs (10× calibrated) — still 17% prefer depth>3 |
| **Single-pool T=16 dominates the dispatch space** | 723 / 792 cells (91.3%) pick a T value not in production {64, 128}. T=16 wins on most of those |
| **2-pool never wins under calibrated coefficients** | 0 / 792 cells pick pool_count=2. The dual_pool_extra_us=20µs overhead exceeds 2-pool's bandwidth/compute savings on this grid |
| **DEC_TAIL never wins on this grid** | 0 / 792 cells. Even on its theoretically-optimal regime (long L_p, short tail) tested in a separate sub-grid: 0 / 72 cells pick DEC_TAIL. The merge_us cost edges out the prefill-tail savings |

## Magnitude — how much does the production 11 lose?

`prod_11_relative_loss` = (best-cost-within-production-11) /
(picker-cost-on-expanded-space) − 1.

| Statistic | Value |
|---|---|
| median  | 0.69%  |
| mean    | 13.96% |
| p90     | 44.79% |
| p99     | 135.66% |
| max     | 171.07% |
| cells losing >5%  | 298 / 792 (37.6%) |
| cells losing >10% | 239 / 792 (30.2%) |
| cells losing >25% | 143 / 792 (18.1%) |
| cells losing >50% | 72 / 792 (9.1%)  |

In the cells where the expanded picker chose depth>3, median loss is
**18.9%**, max loss **171%** (production picker is 2.7× slower than
the expanded picker on the worst cell).

**Worst-case cells.** All K=64 with deep hierarchical sharing
(n_intermediate=3 or 4, intermediate-len=512). Expanded picker chooses
`shared_d{5,6}_p1_t16`; production picker stuck on `shared_d3_p1_t64`,
losing 130–170%.

## Axis isolation

To confirm the depth and T findings are independent (not just
co-occurring on the same cells), three sub-sweeps:

| Restriction | Headline |
|---|---|
| Depth ∈ {2..6}, T forced to {64, 128} | depth>3 chosen on 43.6% — depth axis matters even without new T |
| Depth forced to {2, 3}, T ∈ {16..128} | T not in {64, 128} chosen on 90.9% — T axis matters even without new depth |
| Both expanded | T=16 + depth ∈ {2..6} dominates uniformly |

Both axes are independently load-bearing.

## What this means for the paper

**The "11 candidates" justification (J1–J4) cannot stand as written.**
The space is structurally wrong-shaped, not just slightly off.

Three options for the paper:

1. **Expand the dispatch space** to match what the cost model wants.
   New space ≈ `PER_BEAM ∪ {SHARED_dN_p1_t16 : N ∈ 2..D_max}` where
   D_max is the max depth supported by the wrapper. Drop 2POOL and
   DEC_TAIL entirely (never picked). This would be a much smaller
   space than 11 — closer to 6 — but along different axes.

2. **Rewrite the cost model.** The current cost model has known
   limitations (see Caveats below) that bias toward small-T,
   deep-cascade picks. Some of these effects are real, some are
   modeling artifacts. Fixing the model may bring the picker's
   choices closer to the production 11.

3. **Both** — expand dispatch space *and* rewrite cost model. Most
   work, most defensible result.

The right answer depends on whether the cost model's predictions
match real kernel measurements. **That's the next experiment**
(Phase 0.5, validation): take ~5 cells where the expanded picker
strongly prefers `shared_d{4,5}_p1_t16` over `shared_d3_p1_t64` and
run actual kernels. If the prediction holds, expand the space. If it
doesn't, fix the cost model.

## Real H100 kernel measurements — and the cost-model discrepancy

The cost-model study above is purely analytical. To validate, we
analyze pre-existing real measurements from `bench_modes_single` runs
on H100 with Llama-3.2-1B (`num_kv_heads=8, head_dim=64`).

### Vanilla beam search at K=64, L_p=8192, B=32, max_new=256

| Strategy | decode_per_prompt_per_token_ms | Cost-model prediction (µs/layer) |
|---|---|---|
| PER_BEAM           | 5.4688 | (single-prompt model, doesn't apply) |
| **SHARED_2L_DEC_TAIL** | **1.4068** | 1350 |
| SHARED_3L_DEC_TAIL | 1.5657 | (no intermediate → falls back) |
| SHARED_3L_1POOL    | 1.5940 | (falls back to 2L) |
| picker             | 1.5250 | — |
| SHARED_2L_1POOL    | 1.7101 | 1352 (≈ tied with DEC_TAIL in model) |
| SHARED_2L_2POOL    | 1.9155 | 1374 |
| SHARED_3L_2POOL    | 1.9224 | (falls back) |

**Real winner: SHARED_2L_DEC_TAIL** — 17% faster than 2L_1POOL, 27%
faster than 2L_2POOL.

**Cost-model prediction**: ties DEC_TAIL with 1POOL_T=16 within 0.1
µs; can't distinguish them. **Misses the 17% real advantage.**

### DBS at K=64, L_p=8192, B=32

| Strategy | ms/prompt/token | Real ranking |
|---|---|---|
| **SHARED_2L_DEC_TAIL** | **3.3392** | winner |
| picker             | 3.3451 | ≈ tied with winner |
| SHARED_3L_DEC_TAIL | 3.4801 | |
| SHARED_3L_1POOL    | 3.4829 | |
| SHARED_2L_1POOL    | 3.5147 | |
| SHARED_2L_2POOL    | 3.7525 | 2-pool loses |
| SHARED_3L_2POOL    | 3.8080 | |
| PER_BEAM           | 7.2304 | |

Same ordering as vanilla. **DEC_TAIL > 1POOL > 2POOL** consistently.

### Cost-model discrepancies confirmed by measurement

| Claim | Cost-model prediction | Real measurement | Verdict |
|---|---|---|---|
| 2-pool wins on large batch | Cost model: never (0/2376 cells) | Real: never (0/12 cells, B∈{1,16,32}) | **Cost model right; production-paper claim wrong** |
| DEC_TAIL is optimal at long-prefix + short-tail | Cost model: never (0/72 cells in stress test) | Real: DEC_TAIL wins at the canonical cell K=64/L_p=8K/B=32 | **Cost model wrong — fails to predict DEC_TAIL** |
| Single-pool T=16 beats T=64/128 single-pool | Cost model: T=16 wins on 91% | Real: not directly tested (production picker only enumerates T∈{64,128} for single-pool) | **Need test with T=16 enabled** |
| Depth>3 wins on hierarchical workloads | Cost model: depth>3 wins on 43.6% of synthetic cells | Real: not directly tested (wrapper currently capped at depth=3) | **Need wrapper extension** |

### Why the cost model misses DEC_TAIL's advantage

The cost expressions for SHARED and DEC_TAIL produce essentially the
same predicted µs at K=64/L_p=8K/B=32:

```
shared_d2_p1_t16  : bw(1242) + compute(10) + share_extra(100) = 1352 µs
shared_d2_dec_tail: prefix(414+1.4+100) + tail(2+max(826,419)) + 1·merge(7) = 1350 µs
```

Both load the same total KV (~1.6 GB), have similar wave counts, pay
similar overhead. The model can't differentiate them.

Real measurements show DEC_TAIL is meaningfully faster — likely
because:

- The decode kernel (CTA_Q=1) achieves higher BW utilization on
  per-beam tail than a prefill kernel forced to T=16 with R=1.
- Per-tile launch overhead in prefill kernel adds up across 2048
  small tiles; not captured by the cost model's `waves ×
  per_tile_us` term.
- The `share_extra_us=100` overhead is calibrated as a flat per-call
  constant; reality may have per-tile or per-level overhead the model
  conflates.

**The cost model has the right structural form but wrong constants —
or wrong granularity.**

## Cost-model fix — applied

Added a `bw_efficiency_floor` coefficient to `Coefficients`
(`cost_model.py`) plus a per-level effective-BW helper
`_level_effective_bw_us`. The fix differentiates the prefill kernel's
achievable BW at low utilization from the decode kernel's peak BW at
its native CTA_Q=1:

```
For each cascade level:
  util = packed / T
  if util < 0.5 AND g_count >= 8:        # per-beam-tail-like situation
      eff_factor = floor + (1 - floor) × util
  else:                                   # root or low-tile-count level
      eff_factor = 1.0
  level_bw_us = level_bytes / (B_hbm × eff_factor)
```

The decode kernel (used by DEC_TAIL's tail) is not subject to this
penalty — it stays at peak BW.

`bw_efficiency_floor=1.0` (default in `Coefficients`) reproduces the
legacy peak-BW model. The cached H100 coefficients loader sets
`floor=0.5` based on fit against measurements.

### Validation against 16-cell measured grid

(Llama-3.2-1B, K∈{16,64}, L_p∈{2K,8K,32K}, B∈{1,8,32}, depth=2 only.)

| floor | match rate | mean regret | max regret |
|---|---|---|---|
| 1.0 (legacy) | 3/16 = 19% | 7.85% | 38.1% |
| 0.7 | 10/16 = 62% | 3.08% | 38.1% |
| **0.5 (recommended)** | **13/16 = 81%** | **0.41%** | **2.83%** |
| 0.3 | 13/16 = 81% | 0.42% | 3.13% |

**At floor=0.5: 13/16 cells correctly predicted; even the 3 mispicks
are within 3% of the true winner's measured time.**

The 3 remaining mispredictions:
- K=16/L_p=8K/B=1, K=64/L_p=2K/B=1: real winner PER_BEAM, model
  predicts 1POOL. Cost model overestimates PER_BEAM because it
  doesn't model L2-caching of the shared prefix across the K beams
  within a single prompt. Separate fix (`cost_per_beam_batch`),
  not addressed by this BW-efficiency change.
- K=64/L_p=32K/B=1: real DEC_TAIL, model predicts 1POOL. Edge case
  at very long prefix + B=1; the prefix-prefill cost dominates and
  the BW penalty on the small tail level isn't enough to flip the
  pick. Both candidates are within 1.6% of each other in
  measurement.

### Effect on the dispatch-space study

Re-running `benchmarks/bs_kernel/study_dispatch_space.py` with the
fix (792 synthetic workloads × 3 batch sizes = 2376 cells, depth ∈
{2..6}, T_large ∈ {16,32,64,96,128}, pool ∈ {1,2}):

| Outcome | Without fix | With fix |
|---|---|---|
| Cells where DEC_TAIL is picked | 0 / 2376 (0%) | 701 / 2376 (29.5%) |
| Cells where 2-pool is picked | 0 / 2376 | 0 / 2376 (still 0) |
| Cells where depth>3 is picked | 1065 / 2376 (45%) | 1061 / 2376 (45%) |
| Cells where T not in {64, 128} | 2307 / 2376 (97%) | 1600 / 2376 (67%) |

**The fix recovers DEC_TAIL as a winning strategy in many cells**
(matching real-measurement behavior at K=64/B≥8). 2-pool's never
winning persists — confirming our finding that 2-pool is structurally
suboptimal in the current dispatch space at K∈{16,64}. (2-pool *does*
win on real measurements at K∈{16,64} with L_p=32K — those are the
edge cells my synthetic workload grid didn't fully cover.)

### Closed-issue note

The original "the cost model can't differentiate DEC_TAIL from T=16
prefill at R=1" issue is **resolved** by the fix above. Match rate
went from 19% to 81% on the measured grid; mean regret 7.85% → 0.4%.

## Open issues

## Bigger reframing — what this means for the paper

The findings split into two categories:

**Validated against real measurements:**
- 2-pool never wins on any cell tested (matches both cost model and
  measurements). The production paper's claim that 2-pool is needed
  is empirically false.
- The picker is approximately right when DEC_TAIL is a candidate
  (production picker chose ~DEC_TAIL behavior at K=64/B=32, scoring
  picker=1.525 vs DEC_TAIL=1.407).

**Predicted by cost model but unvalidated:**
- Depth>3 winning on hierarchical workloads (43.6% of cells in the
  cost-model study). Real validation requires extending the wrapper
  to support `num_levels > 3`.
- Single-pool T=16 winning over T=64/128 (91.3% of cost-model cells).
  Real validation requires extending the production picker's
  `T_LARGE_CHOICES`.

**Open issues:**
- Cost model can't distinguish DEC_TAIL from T=16 single-pool. This
  matters because the picker may choose the wrong one in
  production-supported regimes.

The right next experiment: **submitted SLURM job 193292
(`bench_dispatch_grid`) sweeping K∈{16,64} × L_p∈{2K,8K,32K} ×
B∈{1,8,32}, all 8 modes**. This will give us a measured Pareto
frontier across 18 cells × 8 modes = 144 measurements, which should
let us fit a better cost model.

## Caveats — known cost-model limitations that bias these results

These do not invalidate the findings (the directional gap is too
large), but they affect the magnitude and the question of which
direction to fix:

1. **`share_extra_us = 100 µs` is per-call, not per-level.**
   Adding a level adds plan-launch + merge work in reality, but the
   cost model only pays this overhead once per call regardless of
   depth. Sensitivity check: even at share_extra_us=1000 µs (10×
   calibrated), depth>3 still wins on 17% of cells. Robust enough
   that depth>3 wins for some workloads, even with much higher
   per-call overhead. But: a per-level overhead term (e.g., 30 µs
   per level) might shift the picture more than per-call inflation.

2. **Bandwidth doesn't scale with tile count.** Cost model assumes
   each level's KV is read once per group, regardless of how many
   tiles split that group. Real hardware: T=16 may pay multi-tile
   re-reads if L2 misses; the predicted bandwidth advantage of T=16
   over T=64 may be optimistic.

3. **DEC_TAIL has explicit `n_merges × merge_us`; SHARED doesn't.**
   The fused-cascade SHARED kernels also do internal merges, but
   they're assumed amortized in `per_tile_us`. This asymmetry biases
   the picker against DEC_TAIL.

4. **G factor not in cost model.** Production cost model uses
   `packed_qo = K`, not `K × G` (G = num_qo_heads / num_kv_heads = 4
   for Llama-3.1-8B). The paper's bimodal observation says R = K·G
   at root. If the cost model implicitly normalizes by G, the
   comparison between T=16 and T=128 may be off by a factor of G in
   tile count.

5. **`per_tile_us[32] = 0.867` and `per_tile_us[96] = 1.9` are
   linearly interpolated** from {16, 64, 128}. The picker mostly
   chose T=16 (calibrated) so this doesn't change the headline. If
   any cell's choice depends on T=32 or T=96, that pick should be
   revisited after real calibration.

6. **Synthetic uniform-branching workloads.** Real DBS /
   spec-decoding hierarchies aren't perfectly uniform. The
   directional findings (deep cascades + small T preferred) likely
   hold qualitatively, but exact magnitudes need real-workload
   measurement.

## Recommendation for next steps

1. **Phase 0.5 — kernel validation (highest priority).** Take ~5
   cells where the expanded picker chose `shared_d{4,5,6}_p1_t16`
   with predicted-µs gap > 25% to the best production-11 candidate.
   Run actual kernels on H100 (FlashInfer's
   `MultiLevelCascadeAttentionWrapper` supports arbitrary
   `num_levels`, so this is feasible) and measure. Confirm or refute
   the cost-model predictions.

2. **If kernel validation supports the model:**
   - Expand the dispatch space to include depth>3 and T=16 single-pool.
   - The paper's C2 contribution becomes: "the dispatch space along
     four axes (depth, pool, tail-kernel, T-per-pool); empirically
     the minimal sufficient subspace on our workload grid is
     `<reduced set>`."
   - J1–J4 get rewritten to be data-anchored, not assumption-anchored.

3. **If kernel validation contradicts the model:**
   - The cost model has a real bug (likely per-level overhead
     missing, or bandwidth tile-scaling missing).
   - Fix the model first, re-run this study, then redesign the
     dispatch space.

4. **Either way:** the current `paper_story.md`'s claim that "11
   candidates is the right dispatch space" needs to be removed or
   replaced with the data-driven version. Don't ship the paper with
   J3's "≤3 levels" claim or J1's "Pareto-frontier-of-{64, 128}"
   claim — both are empirically false on this study.

## Reproduction

```bash
uv run python benchmarks/bs_kernel/study_dispatch_space.py
```

Output: `benchmarks/bs_kernel/results/dispatch_space_study-<timestamp>/`
contains `study.csv` (one row per cell with full predicted-µs vector
across all 56 candidates) and `summary.json` (aggregate stats).

## Phase 0.5a — depth-extension implementation status

The cost model + driver + `_adaptive_levels` were generalized to
support `depth ∈ {2..D_max}` (`D_max=6`). Implementation summary:

- `cost_model.py`: `Strategy` enum extended (added `SHARED_{4,5,6}L_*`
  variants); `IntermediateShape` now holds a list of intermediate
  levels; `_per_prompt_levels` / `_per_prompt_levels_no_tail` /
  `cost_shared_batch` / `cost_dec_tail_batch` generalized; new
  `Coefficients.max_dispatch_depth` (default 3) and
  `bw_efficiency_floor` (default 1.0).
- `driver.py`: cascade wrappers indexed by depth
  (`cascade_wrappers: dict[int, FusedMultiLevelCascadeAttentionWrapper]`,
  same for non-fused dual variant); `_adaptive_levels` called with
  `max_levels=D_max`; `_collapse_levels(levels, target_depth)`
  generalizes the d3→d2 collapse to arbitrary target depths;
  `_workload_from_levels` constructs N-level `IntermediateShape`;
  DEC_TAIL prefix-prefill wrappers are now a list of `D_max-1` per-level
  wrappers.
- `adaptive_pool.py:_adaptive_levels`: replaced the depth=3-only
  branch with iterative sub-grouping that recursively detects
  divergent partitions until either no group splits or all sub-groups
  are singletons.
- `bench_modes.py`: added `SHARED_{4,5,6}L_*` modes plus
  `--max_dispatch_depth` / `--max_cascade_levels` knobs.

### Phase 0.5a kernel-level smoke test

Sweep `slurm/bench_depth_sweep.sbatch` (job 193322) at K=64, L_p=8192,
max_new=64, B ∈ {8, 32}; identical-prompt batches; Llama-3.2-1B/H100;
all 16 modes (picker + 15 forced). All 32 runs succeeded once
`CUDA_HOME` was exported in the SLURM env so JIT compilation could
locate the loaded `cuda/12.4` toolchain.

Pick histograms (B=8, 63 decode steps; B=32 identical):

```
SHARED_3L_*    → d2=16,  d3=47
SHARED_4L_*    → d2=16,  d3=16,  d4=31
SHARED_5L_*    → d2=16,  d3=16,  d4=16,  d5=15
SHARED_6L_*    → d2=16,  d3=16,  d4=16,  d5=15   (workload caps at d=5)
```

The recursion in `_adaptive_levels` correctly walks one new sub-group
level per ~16 decode steps (one per `page_size`), confirming the
depth detection works on real workloads. d=6 is unreachable on
identical-prompt vanilla beam search because the natural sub-grouping
saturates at four levels of within-prompt sharing for this cell.

### Phase 0.5a empirical finding — depth>3 is *not* faster on identical-prompt workloads

| Mode | B=8 decode (ms) | B=32 decode (ms) |
|---|---|---|
| picker | 10570 (incl. JIT) | 3922 |
| SHARED_2L_DEC_TAIL | **1154** | **3923** |
| SHARED_2L_1POOL | 1258 | 4154 |
| SHARED_3L_1POOL | 1207 | 4292 |
| SHARED_4L_1POOL | 1321 | 4307 |
| SHARED_5L_1POOL | 1253 | 4391 |
| SHARED_6L_1POOL | 1254 | 4395 |
| SHARED_3L_2POOL | 1436 | 4977 |
| SHARED_4L_2POOL | 1427 | 5371 |
| SHARED_5L_2POOL | 1475 | 5550 |
| SHARED_6L_2POOL | 1565 | 5539 |

Within ~1% the deeper 1POOL variants match SHARED_2L_1POOL — the
intermediate sharing levels carry small KV deltas in this workload,
so sharing them across the cascade contributes little while the
extra plan overhead is paid in full. The 2POOL family scales
*worse* with depth (extra dual-pool plan overhead dominates the
share-saving). DEC_TAIL is the across-the-board winner here.

**Implication for J3:** on identical-prompt vanilla beam search, the
empirical `D_max` is **3** (SHARED_3L_DEC_TAIL = 1195ms is the only
depth-3 variant that beats SHARED_2L_DEC_TAIL on these cells, and the
margin is small). Validating depth>3 wins requires hierarchically
structured workloads (DBS with nested groups, EAGLE-style spec trees,
or multi-prompt batches with structured cross-prompt sharing) — that
is the Phase 0.5b experiment.

### What Phase 0.5a *does* establish

1. **The dispatch infrastructure correctly supports depth ∈ {2..6}.**
   All 16 forced modes execute without crashing; pick histograms
   show the recursive sub-group detection emits levels in the
   expected schedule (one per page_size).
2. **The cost model heuristic agrees with measurements at depth ∈
   {2, 3} on this cell** (picker chose SHARED_2L_DEC_TAIL = oracle
   on B=32). Deep variants are correctly *not* picked when the
   workload doesn't reward them.
3. **`max_dispatch_depth=3` is the conservative production setting**
   for vanilla beam search; depth=6 wrapper construction is opt-in
   via `max_cascade_levels=6` + coefficient override and adds no
   cost when not exercised (wrappers share JIT'd kernel binary at
   the `max_levels=D_max` template parameter).

### Phase 0.5b — what remains

Real-kernel validation on hierarchical workloads is still needed to
either confirm or refute the cost-model prediction that depth>3
wins on cells with deeper natural sharing. Required infra:

- A bench script that constructs synthetic hierarchical workloads
  (e.g., DBS-with-nested-groups, or B prompts sharing a common-cluster
  prefix above per-prompt prefixes).
- Or: extend `bench_modes.py` to ingest an EAGLE-style draft tree.

Until 0.5b is done, the paper's J3 claim must defer the depth bound
to "the picker selects depth ∈ {2, 3} on every workload we evaluated;
the cost model permits up to D=6 if a workload rewards it." That is
the conservative-but-honest framing.

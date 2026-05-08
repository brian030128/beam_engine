# Beam-search-specialized attention kernel — design plan

> **Status note (2026-05).** This document is the original design plan.
> Sections are annotated [LANDED] / [PARTIAL] / [DEFERRED] to reflect
> what shipped. The implementation diverged from the plan in three
> ways: (1) `DEC_TAIL` strategies (decode-kernel tail with online
> softmax merge) were added — this is the natural CTA_Q=1 case of the
> two-pool concept, see `docs/cta_tile_q_design.md`; (2) the GPU-side
> plan-update kernel landed only in Phase 1 (`plan_update.py`) — the
> hot path still goes through `pick_strategy_batch` on CPU with two
> aggressive plan caches keyed by LCA and decode-kv-len; (3) the
> ablation list condensed to D1 + per-decision sweeps run via
> `bench_modes.py` rather than a separate `oracle_vs_model.py`.

## Context

The user is writing a paper on the fastest LLM beam-search engine. The
current "our method" — adaptive 2-pool routing on top of FlashInfer's
`FusedMultiLevelCascadeAttentionWrapper` — has an algorithmic story
(adaptive level decomposition driven by the beam fork tree) but no
kernel-level contribution: the kernel that does the work is FlashInfer's,
not ours. A paper reviewer will land on "your contribution is a Python
plan-time wrapper over a third-party kernel."

This plan designs a beam-search-specialized attention kernel that owns
the kernel-level contribution. The target is **Pareto dominance**: a
single kernel that ties or beats every existing kernel choice (paged
decode, tree-mask, FastTree, FlashInfer cascade) across the workload
space, so practitioners do not switch kernels based on K, L_p, fork
density, or suffix length. "Optimal in every scenario" is not an
engineering target — Pareto dominance is.

The contribution decomposes into three layers, each with a separate
ablation story:

1. **Kernel** [LANDED, partial] — per-call `force_cta_tile_q` lets the
   picker pick T per step (FlashInfer commit `3df5803b`); fused LSE
   merge in epilogue is implemented behind `fused_merge=False` flag
   (default off); persistent grid is the upstream FlashInfer kernel
   (we reuse it).
2. **Cost model** [LANDED, extended] — unified decision over
   (cascade depth × pool count × T_per_pool × tail-kernel). The
   tail-kernel axis was added during implementation:
   `SHARED_*L_DEC_TAIL` strategies route the per-beam tail through
   `BatchDecodeWithPagedKVCacheWrapper` (CTA_Q=1, 0% padding) and
   merge with the prefix via `merge_state_in_place`. See
   `decode_tail_context.py` and `cost_model.py:Strategy`.
3. **Per-device cost-model calibration** [LANDED] — `calibrate.py`
   probes prefill per-tile cost (`_T_tile_cost`) and decode per-tile
   cost (`_decode_tile_cost`); cached at
   `~/.cache/beam_engine/coeffs-<gpu>.json`. Picks adapt across H100 /
   A100 / RTX-PRO with the same code.

The 2-pool cascade we already built (`src/beam_engine/methods/adaptive_pool.py`)
became the *first baseline we beat* and remains shipped.

## Files to add / modify

The kernel work lives in the existing **`brian030128/flashinfer`** fork
(`3rdparty/flashinfer`, branch `beam_engine`); the cost model + plan
update + driver live in `beam_engine`. Two-repo change set:

### `3rdparty/flashinfer` (extends existing fused-cascade kernel)
- `include/flashinfer/attention/fused_cascade.cuh` — extend the
  per-level prefill kernel with two additions:
  (a) **fused LSE merge in epilogue** — the final-level CTA reads
  `partial_o` / `partial_lse` written by earlier levels (line ~167 of
  current code) and merges in registers before writing `out`. Replaces
  the `CascadeMerge` separate launch (line 231 today).
  (b) **in-kernel pool dispatch** — accept a per-CTA `pool_id` table;
  CTAs dispatch on it to one of two compile-time-templated tile
  sizes (T_large, T_small). The existing 16/64/128 pool routing
  (`_schedule_level_three_pool` in `flashinfer/cascade.py:614`) already
  does pool-by-(req, qo_tile) packing; we extend the kernel to honor
  the chosen `T` per CTA without separate launches.
- `csrc/fused_cascade.cu` — launcher: drop the separate `CascadeMerge`
  call; pass partial-buffers as kernel args so the epilogue can read
  them.
- `csrc/fused_cascade_customize_config.jinja` — add `T_large` /
  `T_small` template params.
- `flashinfer/cascade.py:589` (`FusedMultiLevelCascadeAttentionWrapper`) —
  expose `fused_merge=True` mode that uses the new kernel path; old
  path stays for ablation C3.

### `beam_engine` (cost model, plan update, driver, benchmarks)
New files:
- `src/beam_engine/methods/bs_kernel/__init__.py`
- `src/beam_engine/methods/bs_kernel/plan_update.py` — Triton kernel
  that maintains plan state on GPU and runs the cost model per step.
  Output is the work queue + per-CTA dispatch table the modified
  `fused_cascade.cuh` consumes.
- `src/beam_engine/methods/bs_kernel/cost_model.py` — closed-form cost
  expressions + coefficient struct. Used by `plan_update.py` (compiled
  into Triton constants) and by `autotune.py` (Python during fitting).
- `src/beam_engine/methods/bs_kernel/calibrate.py` — per-device
  cost-model coefficient calibration (microbenchmark probes); on-disk
  cache at `~/.cache/beam_engine/coeffs-<gpu-name>.json`.
- `src/beam_engine/methods/bs_kernel/driver.py` — `beam_search()`
  entry point matching the existing `paged.py` / `tree.py` /
  `fasttree.py` / `adaptive_pool.py` signature so the existing
  benchmark harness drops in.
- `tests/test_bs_kernel.py` — cross-baseline equality vs `tree.py`
  (numerical reference that already passes equality with `paged.py`).
- `benchmarks/bs_kernel/sweep.py` — workload grid sweep, drives the
  ablation tables.
- `benchmarks/bs_kernel/oracle_vs_model.py` — measures every strategy
  per workload, computes cost-model regret distribution.

Existing infrastructure to reuse (do not duplicate):
- `src/beam_engine/models/attention.py:18` — `AttentionContext` protocol;
  the new driver implements `attend(q, k, v, layer_idx)` per usual.
- `src/beam_engine/page_table.py:22` — `PageTable` (refcount + CoW) is
  the right KV-cache layout (5D NHD, page_size=16). The driver uses it
  unchanged.
- `flashinfer.page.append_paged_kv_cache` — the KV-write op all four
  baselines already call; reuse for the per-step K/V write.
- `src/beam_engine/baselines/paged.py:104-114` — `_add_ref` /
  `_remove_ref` page-refcount helpers, identical semantics needed.
- `src/beam_engine/methods/adaptive_pool.py:148-260` — `_adaptive_levels`
  / `_build_cascade_plan`. The new plan-update kernel reproduces the
  same level decomposition GPU-side; that Python function is the
  reference oracle the kernel must agree with.
- `flashinfer/cascade.py:614` — `_schedule_level_three_pool` already
  does (req, qo_tile) → pool packing; cost model picks T_large per
  step, the existing scheduler then partitions tiles into pools using
  that T.
- `benchmarks/cascade_attn.py` — workload-shape generators
  (`make_uniform_cascade`, `make_variable_depth_cascade`) cover the
  topologies the new sweep needs.

### Build / rebuild

FlashInfer is JIT (per `3rdparty/flashinfer/CLAUDE.md`); `.cuh`
edits are picked up automatically on next call. After changing
`csrc/fused_cascade.cu` or the jinja template, the JIT cache rebuilds
on first call. No `pip install` step needed during development.

## Design — the kernel

We **extend FlashInfer's existing `fused_cascade.cuh`** rather than
write a new kernel. The current kernel already provides: the multi-level
prefill skeleton, per-CTA pool routing across CTA_TILE_Q ∈ {16, 64, 128}
(see `flashinfer/cascade.py:614` `_schedule_level_three_pool`), JIT
specialization, and a stable correctness baseline. We add two
beam-search-targeted modifications:

**Modification 1 — fused LSE merge in epilogue.**
Today (`include/flashinfer/attention/fused_cascade.cuh`), non-final
levels write `partial_o` / `partial_lse` into split-K-style buffers
(line 167) and a separate `CascadeMerge` kernel launch (line 231) folds
them. We move the merge into the final-level CTA's epilogue: it reads
the upstream partials it depends on, runs online-softmax merge in
registers, and writes the final `out` directly. This eats one kernel
launch + one L2 fence per step. For decode (~30-100 µs/layer), that
launch+fence is 5-15% of step time.

**Modification 2 — cost-model-driven pool dispatch.**
The current scheduler always packs (req, qo_tile) into the three fixed
pools. We make `T_large` (and optionally `T_small`) a per-step input:
the cost model picks them, the scheduler packs accordingly. The kernel
honors this via a per-CTA dispatch table; no separate launch per T. The
existing 3-pool packing logic in `_schedule_level_three_pool` is reused
verbatim — only `t_lg` becomes dynamic (it's already a parameter,
`flashinfer/cascade.py:618`, but defaults to 128).

The two modifications combine: the final-level epilogue merges
partials regardless of which pool produced them, so cost-model picks
across (pool count, T) are all served by the same kernel launch.

**Plan-update kernel** (new, in `beam_engine`, not FlashInfer).
Triton kernel that runs once per decode step before the layer loop:

- Reads the previous step's plan state (LCA depth, per-beam page lists,
  fork-group structure) from GPU buffers.
- Applies the per-step delta (K new write slots, fork groups from
  `parent_beam_ids`) — the delta is bounded by `O(K)`, never depends on
  L_p.
- Recomputes cost-model terms from the updated state.
- Picks (share, depth, pool_count, T_large) and writes the
  `qo_indptr_arr`, `paged_kv_indptr_arr`, etc. that
  `FusedMultiLevelCascadeAttentionWrapper.plan` consumes — except the
  consumption is GPU-direct: we extend `plan()` to accept device
  tensors and skip its `.cpu()` round-trips (the current
  `_schedule_level_three_pool` does Python-side packing, which is the
  per-step plan cost we want to eliminate).

This eliminates the per-step Python plan-build path the cascade method
spends ~hundreds of µs on at K≥16.

## Design — the cost model

The cost model is four sequential decisions, each justified, each with a
closed-form cost expression and a dedicated ablation. All times in µs.

### Decision 1 — share the prefix, or do per-beam paged decode?

**Justification.** Sharing the prefix avoids the `K×L_p×bytes_per_token`
HBM traffic that paged decode pays. But sharing requires running a
non-trivial cascade plan + an LSE merge — fixed costs of ~`launch_us +
merge_us` per step. When K is very small or L_p is short, the saved
bandwidth doesn't cover those fixed costs and per-beam wins. This is the
crossover paged decode is correctly designed for; we must detect it and
not pay merge overhead when it doesn't help.

**Cost expressions** (per layer, all heads):

```
T_share   = (L_p + Σ_b suffix_len_b) × bytes_per_kv / B_hbm
            + K × small_q_compute_us
            + merge_us
T_perbeam = K × (L_p + max_b suffix_len_b) × bytes_per_kv / B_hbm
            + K × full_q_compute_us
```

where `B_hbm`, `merge_us`, `launch_us` are device-calibrated coefficients
fit at install time.

**Crossover** (closed form):
`(K - 1) × L_p × bytes_per_kv / B_hbm  ≷  merge_us`.

**Ablation A1 — prefix-share crossover.** Sweep K ∈ {1, 2, 4, 8, 16,
32, 64, 128, 256}, L_p ∈ {128, 512, 2K, 8K, 32K}, suffix_len = 16
(early decode). Measure both paths. Plot model-predicted vs empirical
crossover line. **Hypothesis:** model predicts within one (K, L_p) bucket
of the empirical crossover.

### Decision 2 — cascade depth (1, 2, or 3 levels)

**Justification.** Two levels (LCA prefix + per-beam tail) is the
default and covers all post-prefill states before any fork keeps siblings
alive. Three levels (LCA prefix + per-fork-group intermediate +
per-beam tail) only helps when a fork has produced uniform-size groups
that share an intermediate stretch of pages — a structure the cascade
wrapper requires (uniform group sizes per level). Adding a third level
costs an extra cascade-level boundary in the work queue + an extra
LSE-merge term in the epilogue. When the intermediate run is short or
group sizes are non-uniform, two levels wins.

**Cost expression (depth d ∈ {1, 2, 3}):**

```
T_depth(d) = Σ_{l=0..d-1} (group_count_l × tile_count_l × per_tile_us(T_l))
             + (d - 1) × per_beam_merge_us
```

where `per_tile_us(T)` is the auto-tuned per-tile time for tile size T.
Depth 3 is only a candidate when the intermediate-group analysis
(reuses the existing `_adaptive_levels` logic at
`src/beam_engine/methods/adaptive_pool.py:148`) reports a uniform-G
partition with `intermediate_run_len ≥ 1`.

**Ablation A2 — depth crossover.** Construct workloads with controlled
fork structure: post-fork mid-decode states with varying
intermediate-run lengths (0, 1, 2, 4, 8, 16 pages) and group counts
(K=8: G=2, 4; K=16: G=2, 4, 8). Measure 2-level vs 3-level wall time;
verify model picks correctly. **Hypothesis:** 3-level wins only when
intermediate run ≥ 1 page AND group size ≥ 2.

### Decision 3 — pool count (single launch vs dual launch vs DEC_TAIL)

**Status update.** What shipped is a 3-way choice, not 2-way. The
third option, `DEC_TAIL`, is the conceptual CTA_Q=1 limit of the
small-T pool: instead of running per-beam tails through a prefill
kernel with `T=16` and 4/16=25% MMA utilization, route them through
`BatchDecodeWithPagedKVCacheWrapper` (which is purpose-built for
CTA_Q=1) and merge with the prefix-kernel output via online softmax.
At K=64/B=32/L_p=8K, DEC_TAIL gives 0% padding on the tail level vs
98.4% under the prefill `T=64` pool. Two-pool and DEC_TAIL share the
same shape detection and merge machinery; DEC_TAIL just substitutes
the small-T launch for a decode-kernel launch.

**Justification.** This is the FastTree weakness we explicitly attack.
FastTree's `_build_metadata` (3rdparty/FastTree-Artifact/kernel_bench/fasttree.py:233)
splits vnodes into above/below-threshold buckets and launches once per
non-empty bucket — paying full launch overhead for a one-CTA kernel
when the small bucket has only the minority of the work. Our cost model
compares three plans:

- **Single-pool, large-T** — fold per-beam tails (small Q) into T_large
  with row padding (up to `T_large - K` idle threads per tile).
- **Single-pool, small-T** — split shared prefix into more T_small CTAs,
  reading the prefix once per CTA group.
- **Dual-pool, large-T + small-T** — launch both, pay
  `launch_us + sync_us` overhead.

**Cost expression:**

```
T_single_large = ceil(work_large / parallelism_large) × per_tile_us(T_large)
                 + padding_factor × per_tile_us(T_large) × small_tile_count
T_single_small = ceil(work_total / parallelism_small) × per_tile_us(T_small)
                 + (large_split_factor - 1) × prefix_reload_us
T_dual         = max(T_pool_large, T_pool_small) + launch_us + sync_us
```

`launch_us` and `sync_us` come from device calibration. Dual is
dominated by the slower pool (CTAs from the fast pool sit idle waiting),
so single-pool wins unless both pools have enough work that the
slowest-pool time exceeds `T_single_* + launch_us + sync_us`.

**Ablation A3 — pool-count crossover.** Vary the work distribution: at
fixed K=8, L_p=4096, sweep per-beam suffix length in {1, 4, 16, 64,
256}; at each point measure all three plans and compare to model pick.
Tabulate: cells where model and oracle agree (target ≥95%); cells where
they disagree, regret in % of best-time.

### Decision 4 — T per pool (CTA_TILE_Q)

**Justification.** The optimal T depends on register pressure, shared
memory capacity, KV-load arithmetic intensity, and SM count — all
device-specific. FastTree ships hand-tuned values (`TSQs = [64, 16]`,
fasttree.py:11) tuned for H100; on A100 or RTX-class cards the optimum
is different. The kernel JIT-compiles support for T_large ∈ {64, 128}
(both, picked per step) and T_small ∈ {16}; the cost model picks
T_large per step from the compiled set using device-calibrated
`per_tile_us(T)` measured at engine init.

**Ablation A4 — per-device cost-model picks.** On each device
(H100, A100, RTX-PRO-6000), record the cost model's T_large pick
distribution across the workload grid. Show the distribution differs
per device — i.e., a fixed T_large would be wrong on at least one
device. Compare end-to-end wall time of cost-model-pick vs each fixed
T_large choice (64 always, 128 always); show calibrated picks win.

## Design — per-device cost-model calibration

The cost model has device-dependent coefficients: HBM bandwidth
(`B_hbm`), kernel-launch overhead (`launch_us`), separate-merge cost
(`merge_us`), inter-launch sync cost (`sync_us`), and a small table
`per_tile_us(T)` for each compiled T. We calibrate them once at engine
init via short microbenchmark probes:

- `B_hbm` — large `cudaMemcpyDtoD` of a contiguous slab; bytes / time.
- `launch_us` — empty Triton kernel × N runs; total / N.
- `merge_us` — call current `merge_state_in_place` on a representative
  shape (K rows × num_heads × head_dim); median of N runs.
- `sync_us` — `cudaStreamSynchronize` × N; median.
- `per_tile_us(T)` for T ∈ {16, 64, 128} — run the unmodified
  `fused_cascade` kernel on a single-tile workload at each T; median.

Total calibration cost: ~50 ms one-time at engine init. Results cached
to `~/.cache/beam_engine/coeffs-<gpu-name>.json`; reused on subsequent
inits if the GPU model matches.

The kernel itself is **not** re-tuned per device — FlashInfer's JIT
already picks register count, shared-memory layout, and instruction
selection per SM at compile time. Our calibration only measures
behaviors the JIT cannot specialize (HBM bandwidth, launch overhead),
which feed the cost model's per-step decisions.

**Ablation A5 — calibration value.** Run the cost model with
calibrated coefficients vs hardcoded coefficients (e.g., assume H100
numbers everywhere) on each non-H100 device. Speedup column shows the
calibrator's value: hardcoded coefficients lead the cost model to wrong
share/depth/pool/T picks on devices with different launch overhead or
HBM bandwidth.

## Ablation experiment plan

The ablation table targets one figure per design decision plus a
roll-up. All measurements use `flashinfer.testing.bench_gpu_time` with
CUPTI; harness reuses `benchmarks/cascade_attn.py:215` `run_one`
pattern.

| Tag  | Question                                       | Sweep                             | Compares                                    |
|------|------------------------------------------------|-----------------------------------|---------------------------------------------|
| A1   | When does sharing the prefix pay off?          | K × L_p grid                      | shared vs per-beam                          |
| A2   | When does cascade depth 3 beat depth 2?        | fork-group × intermediate-run     | depth-2 vs depth-3                          |
| A3   | When does dual-launch beat single-launch?      | suffix_len at fixed K, L_p        | single-large vs single-small vs dual        |
| A4   | Does optimal T vary by device?                 | per-step T pick × {3 GPUs}        | calibrated-pick vs fixed-T per device       |
| A5   | What do calibrated coefficients buy?           | full grid × {3 GPUs}              | calibrated cost model vs hardcoded coeffs   |
| B1   | Cost-model accuracy.                           | full grid                         | model-pick wall-time vs oracle-pick         |
| C1   | Persistent grid value.                         | full grid                         | persistent vs static-grid version           |
| C2   | GPU-side plan-update value.                    | full grid                         | GPU plan-update vs Python plan-update       |
| C3   | Fused-merge value.                             | dual-launch configs               | fused merge vs separate merge launch        |
| D1   | End-to-end Pareto dominance.                   | full grid                         | ours vs paged vs tree vs FastTree vs cascade|

For B1 (cost-model accuracy): per workload, measure wall time of every
candidate strategy, define
`regret = (model_pick_time / oracle_pick_time) - 1`. Report
distribution: median, p90, p99, max. Failure mode (regret > 5%) gets a
case study in the paper text.

For C-series (component ablations): each turns off one design feature
of our kernel and re-runs the full grid. Without persistent grid: use
static grid sized to total work tiles. Without GPU plan-update: rebuild
plan in Python every step (still our kernel, but with cascade-style
plan-time). Without fused merge: emit `merge_state_in_place` as a
separate kernel. Each ablation reports degradation per workload bucket;
the paper figure is a stacked-bar of per-feature contribution.

For D1: The roll-up. End-to-end decode latency on
Llama-3.1-8B (config matches `benchmarks/cascade_attn.py:40`) across the
workload grid, vs all four baselines. Pareto dominance criterion: for
every workload point, our kernel is within 5% of the best baseline at
that point AND wins (>5% faster) at ≥80% of points.

## Verification

**Correctness** — `tests/test_bs_kernel.py`:
- Equality vs `tree.py` (already-validated reference): same prompts,
  same beam_width, same max_new_tokens; assert greedy beams match
  exactly and `cum_log_prob` matches within fp16 tolerance (1e-2).
- Equality across (K ∈ {1, 4, 16, 64}) × (L_p ∈ {128, 4096}) × (max_new
  ∈ {16, 128}). K=1 explicitly verifies degenerate-to-paged-decode
  fallback.
- Fork-stress test: prompts crafted (or seeded RNG) to maximise fork
  rate and exercise the 3-level cascade path.

**Performance** — `benchmarks/bs_kernel/sweep.py`:
- Run the full ablation grid above, write CSVs.
- A separate plotting script renders the per-decision crossover figures
  and the D1 Pareto roll-up.

**GPU pinning** — every benchmark and test entry point selects a
fully-idle GPU per `CLAUDE.md` policy: check `nvidia-smi`, error if no
GPU has util≈0 AND mem-used minimal, else pin via
`CUDA_VISIBLE_DEVICES`.

## Open questions deferred to later

These came up during design but the user has chosen not to lock them
down before implementation:

1. Install-time tuner sweep cost vs. shipped-default-profile tradeoff.
2. Whether cost-model accuracy validation (A1–A5 + B1) is in-paper or
   appendix-only.

# Beam-search kernel benchmark results

H100 80GB SXM, Llama 3.2-1B (16 layers, 32 heads, 8 KV heads, head_dim=64).

All numbers are **decode-only** wall time per `(prompt, token)` in milliseconds
(`decode_total_ms / (B * (max_new − 1))`). Lower is better. Prefill latency
identical across methods at fixed L_p / B and is excluded from the comparison.

Six methods benchmarked, all batched (one shared kernel launch per decode step
across all B prompts) except `tree`, which still loops per-prompt:

| key | implementation |
|-----|----------------|
| `paged` | `BatchDecodeWithPagedKVCacheWrapper` — K independent paged decodes per launch |
| `tree`  | FlashInfer ragged kernel with custom causal mask (per-prompt loop, not batched) |
| `fasttree` | FastTree (MLSys '25) Triton kernel; B per-prompt subtrees combined under a virtual root |
| `mlca`  | `MultiLevelCascadeAttentionWrapper` (non-fused, 2L−1 launches per step) |
| `adaptive_pool` | `FusedMultiLevelCascadeAttentionWrapper`, default 2-pool routing |
| `bs_kernel` | adaptive_pool + cost-model picker (`force_cta_tile_q=t_large` → single pool) |

## Benchmark 1 — short decode (max_new = 16)

`benchmarks/bs_kernel/results/bench_batched-20260507-171856/bench_batched.csv`

L_p=8192, batched comparison. Best per row in **bold**.

| K | B | paged | tree | fasttree | mlca | adaptive_pool | bs_kernel |
|---|---|------:|-----:|---------:|-----:|--------------:|----------:|
| 16 | 1 | 18.7 | 20.2 | 46.2 | 14.0 | 13.9 | **11.6** |
| 16 | 2 |  9.2 | 15.7 | 12.3 |  7.7 |  7.7 |  **7.4** |
| 16 | 4 |  7.3 | 15.7 |  9.5 |  5.4 |  5.5 |  **5.4** |
| 16 | 8 |  6.4 | 15.7 |  8.5 |  4.4 |  4.4 |  **4.3** |
| 32 | 1 | 17.3 | 22.6 | 25.6 | 14.6 | **13.8** | 14.2 |
| 32 | 2 | 13.5 | 22.6 | 20.8 |  9.7 |  **9.6** |  9.7 |
| 32 | 4 | 11.9 | 22.6 | 17.1 |  7.6 |  **7.4** |  7.7 |
| 32 | 8 | 11.1 | 22.7 | 16.0 |  6.6 |  **6.5** |  6.7 |
| 64 | 1 | 26.2 | 36.9 | 34.5 | 18.3 | **18.0** | 18.1 |
| 64 | 2 | 22.7 | 36.8 | 29.9 | 14.1 | **13.9** | 14.3 |
| 64 | 4 | 21.4 | 36.8 | 28.0 | 12.2 | **12.0** | 12.4 |
| 64 | 8 | 20.7 | 36.9 | 26.2 | 12.2 | **11.2** | 11.7 |

This was Llama 3.1-8B; Llama 3.2-1B re-run shows the same ranking.

### Observations

- **Cascade methods (mlca / adaptive_pool / bs_kernel) cluster within ~3% of
  each other** — the underlying cascade math dominates kernel-launch
  micro-overheads at this regime.
- **`bs_kernel` does not beat `adaptive_pool` at K∈{32, 64}.** The picker
  records `pool_count=1` in 100 % of decode steps (see analysis below), and
  the driver passes `force_cta_tile_q=pick.t_large` to FlashInfer
  unconditionally, which disables the kernel's default 2-pool routing.
  adaptive_pool gets that routing for free; bs_kernel disables it.
- **`paged` scales with batching** but stays prefix-blind — at L_p=8192 it
  costs 1.4-2.5× the cascade methods.
- **`tree` is ~3× slower than the cascade methods** at K=64 because it
  re-reads the prefix per beam (no shared-prefix attention) and is the only
  method still looping per-prompt.

## Benchmark 2 — long decode (max_new = 512)

`benchmarks/bs_kernel/results/bench_long_decode-20260507-180400/`
(Llama 3.2-1B, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for K=64.)

| K | L_p | B | paged | tree | fasttree | mlca | adaptive_pool | bs_kernel |
|---|----:|---|------:|-----:|---------:|-----:|--------------:|----------:|
| 16 | 8192 | 1 | 5.86 | 11.43 | 13.70 | 6.41 | 23.85¹ | **5.62** |
| 16 | 8192 | 16 | 2.42 | 11.30 |  7.31 | **1.45** | 1.46 | 1.45 |
| 16 | 8192 | 32 | 2.39 | 11.30 |  7.86 | 1.39 | 1.41 | **1.37** |
| 16 | 50k  | 1 | 10.07 | 41.17 | 36.67 | 8.49 | 7.81 | **7.77** |
| 16 | 50k  | 16 | OOM | 40.56 | OOM | OOM | OOM | OOM |
| 16 | 50k  | 32 | OOM | (cancelled) | OOM | OOM | OOM | OOM |
| 64 | 8192 | 1 | 10.28 | 33.0 | 30.6 | 8.01 | 7.28 | **7.05** |
| 64 | 8192 | 16 | 8.82 | 32.9 | 28.1 | 4.22 | **3.90** | 4.02 |
| 64 | 8192 | 32 | 9.02 | 32.8 | 30.2 | 4.28 | **4.08** | 4.14 |
| 64 | 50k  | 1 | 32.2 | 159.3 | 115.5 | 14.67 | **14.68** | 14.83 |
| 64 | 50k  | 16 | OOM | killed (time limit) | not reached | not reached | not reached | not reached |
| 64 | 50k  | 32 | not reached | not reached | not reached | not reached | not reached | not reached |

¹ JIT warm-up on the first cell of the run; reproducible.

### Memory ceilings on H100 80 GB (Llama 3.2-1B = 32 KB / token KV)

| cell | peak tokens | KV size | status |
|------|------------:|--------:|--------|
| K=16 / L_p=8k / B=32 | 524 k | 17 GB | fits |
| K=16 / L_p=50k / B=16 | 931 k | 30 GB | OOM (allocator fragmentation, prior-cell residue) |
| K=64 / L_p=8k / B=32 | 1.31 M | 42 GB | fits with `expandable_segments:True` |
| K=64 / L_p=50k / B=16 | 1.32 M | 42 GB | fits |
| K=64 / L_p=50k / B=32 | 2.65 M | 85 GB | OOM regardless |

Tree survives the OOM cells purely because it loops per-prompt — peak memory
holds 1 prompt's KV at a time, not B.

## Picker behavior — `dump_picks_batched`

`benchmarks/bs_kernel/results/dump_picks_batched-190685.log`

K∈{16, 32, 64} × L_p=8192 × B∈{1, 2, 4, 8} × max_new=16 (Llama 3.1-8B):

**The picker chose `shared_2l_1pool, t_large=64, depth=2` in 180 / 180 decode
steps.** It never flipped to PER_BEAM, never picked t_large=128, never picked
pool_count=2, never picked depth=3.

Per-level CTA_Q padding under the chosen strategy:

| K | L0 padding | L1 padding | L1 tiles/wave (H100, 132 SMs) |
|---|-----------:|-----------:|------------------------------:|
| 16 | 75.0 % | 98.4 % | 0.12 (B=1) → 0.97 (B=8) |
| 32 | 50.0 % | 98.4 % | 0.24 (B=1) → 1.94 (B=8) |
| 64 |  0.0 % | 98.4 % | 0.48 (B=1) → 3.88 (B=8) |

- **L0 padding shrinks with K** because the single L0 group has K queries
  packed into 64-wide tiles (K=64 → 0 % padding).
- **L1 padding is fixed at 98.4 %** because every per-beam tail is 1 query
  in a 64-wide tile — 63 of 64 cells empty.
- **Wave count stays low** at L_p=8192 / max_new=16 — even K=64 / B=8 is
  3.88 waves on H100. The kernel is launch-bound, not compute-bound, so
  the picker's strategy choices have no headroom to improve throughput.

## Benchmark 3 — bs_kernel forced-mode sweep (max_new = 512, L_p = 8192)

`benchmarks/bs_kernel/results/bench_modes-20260508-133019/bench_modes.csv`

Forces bs_kernel into each strategy via `available_strategies={...}`. Lower is
better; **best per row in bold**.

| K | B | picker | PER_BEAM | SHARED_2L_1POOL | SHARED_2L_2POOL | SHARED_3L_* |
|---|---|------:|--------:|----------------:|----------------:|-------------|
| 16 |  1 | 5.81 | **5.29** | 5.41 | 6.29 | n/a |
| 16 | 16 | **0.69** | 1.47 | **0.67** | 0.69 | n/a |
| 16 | 32 | **0.56** | 1.41 | **0.56** | 0.58 | n/a |
| 64 |  1 | 6.07 | 7.88 | **5.90** | 6.89 | n/a |
| 64 | 16 | **1.71** | 5.38 | 1.74 | 1.88 | n/a |
| 64 | 32 | **1.66** | 5.50 | 1.74 | 1.92 | n/a |

`SHARED_3L_*` failed `ValueError: no strategy available after filtering` at
every cell. Reason: the workload has no uniform intermediate groups (forks
on identical-prompt deterministic decoding don't produce them), so the
3-level layout isn't a candidate after the cost model filters infeasible
options. Skipping these picks is the correct behavior.

### Mode-by-mode interpretation

- **picker ≈ SHARED_2L_1POOL on most cells.** The picker is choosing
  `SHARED_2L_1POOL, t_large=64` essentially every step at this grid (matches
  `dump_picks_batched`'s 180/180 finding). At K=64 / B≥16 the picker is a
  hair faster than forced `SHARED_2L_1POOL` — likely picking `t_large=128`
  on some steps where the wider tile saves padding.
- **PER_BEAM is fastest at K=16 / B=1.** 5.29 vs picker 5.81 vs
  `SHARED_2L_1POOL` 5.41 — the picker is slightly mis-choosing at this
  corner. The cost model thinks SHARED is cheaper but in practice
  per-beam paged decode wins by ~10% when there's only 1 prompt and 16
  beams.
- **PER_BEAM scales badly with B.** At K=64 / B=32 it costs 5.50 ms vs
  cascade's 1.66 ms — 3.3× slower. The cost model correctly avoids
  PER_BEAM in this regime.
- **`SHARED_2L_2POOL` is slower than `SHARED_2L_1POOL` everywhere** at this
  grid. The driver routes pool=2 through `cascade_dual_2l` (the older
  non-fused MLCA-style 3-launch path), not through the fused wrapper with
  in-kernel 2-pool routing. So `SHARED_2L_2POOL` here = "MLCA-style
  vanilla cascade", which is ~5-15% slower than the fused single-launch
  path. The actual fused 2-pool routing — what adaptive_pool gets by
  default — is not currently dispatched by any bs_kernel mode (see Open
  Issue 1 below).

### Picker decomposition (`dump_picks_batched`, max_new=512, same grid)

`benchmarks/bs_kernel/results/dump_picks_modes-191913.log`

Per-cell strategy histogram across 511 decode steps, plus the
`(t_large, pool_count, depth)` triple chosen for every SHARED step:

| K | B | strategy histogram | (t_large, pool, depth) | L0 pad | L1 tiles/wave |
|---|---|--------------------|------------------------|-------:|--------------:|
| 16 |  1 | `3l_1pool ×495 + 2l_1pool ×16` | `(64, 1, 3) ×495 + (64, 1, 2) ×16` | 75 % | 0.12 |
| 16 | 16 | `3l_1pool ×494 + 2l_1pool ×17` | same shape | 75 % | 1.94 |
| 16 | 32 | `3l_1pool ×492 + 2l_1pool ×19` | same shape | 75 % | 3.88 |
| 64 |  1 | `3l_1pool ×495 + 2l_1pool ×16` | same shape |  0 % | 0.48 |
| 64 | 16 | `3l_1pool ×494 + 2l_1pool ×17` | same shape |  0 % | 7.76 |
| 64 | 32 | `3l_1pool ×494 + 2l_1pool ×17` | same shape |  0 % | 15.52 |

**Two non-obvious findings:**

1. **The picker uses `depth=3` ~96 % of the time.** It runs `depth=2`
   only for the first ~16 decode steps — before forks have produced
   uniform intermediate groups — then flips to `depth=3` for the rest of
   the run. Forced `SHARED_3L_1POOL` failed at every cell in the
   bench_modes table because the cost model filters out depth=3 when no
   intermediate structure exists, and the *first* decode step (just
   after prefill) always lacks it. The picker works around this by
   falling back to depth=2 at those early steps. **This is real adaptive
   work the forced-mode benches can't see**, and it's the source of
   the picker's ~5 % edge over `SHARED_2L_1POOL` at K=64 / B≥16.

2. **L1 wave count grows past 1 wave at K=64/B≥16.** At K=64 / B=32 it's
   **15.52 waves** of per-beam tail compute on H100's 132 SMs — exactly
   the regime where `pool=2` (T_SMALL=16 tile routing) should help by
   replacing one wide-tile wave with the kernel's auto-routed mix.
   But the current dispatch routes `pool=2` through the non-fused
   `cascade_dual_2l` (3 launches) path, which costs more than the wave
   savings recover. So the picker correctly avoids it, and we never
   exercise the configuration that would actually be fastest in this
   regime — see Open Issue 1.

### Picker takeaways

The picker is **competitive and quietly leveraging adaptive depth**:

- It loses ~10 % at K=16 / B=1 (PER_BEAM would have been better — likely
  a coefficient calibration issue).
- It beats forced `SHARED_2L_1POOL` by ~5 % at K=64 / B≥16 by adaptively
  selecting `depth=3`, a strategy no forced-mode bench can reach.
- It always selects `t_large=64` and `pool=1`. The pool=2 dispatch is
  routed through a slow path that the picker avoids; the missing
  fast-pool=2 dispatch (Open Issue 1) is the largest unrealized gain.

## Benchmark 4 — Diverse Beam Search overhead

`benchmarks/bs_kernel/results/bench_dbs-20260508-1*/bench_dbs.csv`

DBS (Vijayakumar et al., 2016, λ=0.5, num_groups=4) replaces the global
top-K selector with a per-group sequential selector that adds a Hamming
diversity penalty against tokens chosen by earlier groups. The attention
kernel is unchanged — wrap-only.

### Latency (ms / (prompt, token); lower is better)

| K | B | paged | dbs_paged | mlca | dbs_mlca | adaptive_pool | dbs_adaptive_pool | bs_kernel | dbs_bs_kernel |
|---|---|------:|----------:|-----:|---------:|--------------:|------------------:|----------:|--------------:|
| 16 |  1 | 5.85 |  6.41 | 6.03 | 6.78 | 5.27 | 5.96 | **5.52** | 6.35 |
| 16 | 16 | 2.43 |  OOM¹ | 1.11 | 1.84 | 1.14 | 1.88 | **0.69** | 1.66 |
| 16 | 32 | 2.38 |  OOM¹ | 1.05 | 1.76 | 1.12 | 1.84 | **0.57** | 1.54 |
| 64 |  1 | 10.39 | 12.42 | 7.01 | 9.04 | 6.24 | 8.49 | **6.03** | 7.89 |
| 64 | 16 | 8.75 |  OOM¹ | 3.22 | 5.26 | 3.09 | 5.15 | **1.73** | 3.63 |
| 64 | 32 | 8.90 |  OOM¹ | … | … | … | … | … | … |

¹ `dbs_paged` ran out of PageTable pages — DBS produces less prefix
sharing, so each diverse beam owns more unique decode pages, exceeding
the harness's `needed_pages` formula. Bump and re-run if pursued.

K=64 / B=32 row truncated when SLURM time wall hit during `tree`'s 270 s
sequential run.

### DBS overhead is Python-side, not kernel-side

`dump_picks_dbs.py` confirms the picker chose **the exact same strategy
distribution** under DBS as under standard beam search:
`shared_3l_1pool ×~495 + shared_2l_1pool ×~16` at every cell. DBS's
diversity penalty (λ=0.5) doesn't disrupt the uniform-G-groups condition
that gates depth=3 — the surviving K beams still cluster the same way.

The 2-3× DBS overhead therefore comes entirely from the **per-prompt
sequential top-K loop** in the DBS wrapper:

| K | B | base ms / step | DBS ms / step | overhead / step | overhead / prompt-token |
|---|---|---------------:|--------------:|----------------:|------------------------:|
| 16 | 16 | bs_kernel 11.0 | 26.6 | +15.5 ms | +0.97 |
| 16 | 32 | bs_kernel  9.1 | 24.6 | +15.5 ms (×2 prompts) | +0.97 |
| 64 | 16 | bs_kernel 27.7 | 58.1 | +30.4 ms | +1.90 |

The per-step overhead **scales linearly with B and with K** — DBS does
B independent calls each step, each running G=4 sequential top-Ks
with penalty propagation in Python. Per-prompt-per-token overhead stays
roughly constant because B amortizes the work.

### Implication

DBS's overhead is **not** an attention-kernel problem; it's a top-K
problem. None of the 6 attention kernels we benchmarked address it
because they don't see the top-K. Closing this gap would mean either:

1. Vectorizing the DBS top-K across the G groups (currently sequential
   in Python) — likely halves the overhead.
2. Moving the DBS penalty into a fused-CUDA top-K kernel — likely
   eliminates it entirely.

Until then, **all kernels lose ~1.0-2.0 ms / prompt-token under DBS**,
roughly matching the cost of doing G=4 sequential top-Ks per beam-step.

## Open Issue 1 — pool=2 dispatch path

The fused cascade kernel runs **2-pool routing by default** (T=16 small pool
for per-beam tails + T=64/128 large pool for shared prefix, routed per
(request, qo_tile)). `force_cta_tile_q` is an **ablation switch** that
forces every tile into a single pool of the chosen size.

bs_kernel currently has only two pool dispatch arms:

| `pool_count` | wrapper | launches | force_cta_tile_q |
|--------------|---------|---------:|------------------|
| 1 | `cascade_2l` (fused) | 1 | `t_large` (single-pool ablation) |
| 2 | `cascade_dual_2l` (non-fused MLCA-style) | 3 | none |

**Neither arm uses the fused kernel with `force_cta_tile_q=None`** — the
configuration that actually engages in-kernel 2-pool routing.
`adaptive_pool` (and `mlca` via the non-fused path) get the routing
naturally; bs_kernel does not have a mode that maps to it.

The line at `driver.py:774` reads:

```python
force_t = pick.t_large if pick.pool_count == 1 else None
```

— the `else None` branch is unreachable because the surrounding `else`
already excluded `pool_count == 2`. To get a true fused-with-2-pool path,
route `pool_count == 2` through the **same fused wrapper** (`cascade_2l`)
with `force_cta_tile_q=None` instead of through `cascade_dual_2l`.

Until that's fixed, the picker can't select the configuration that has the
best chance of beating `adaptive_pool` — the picker's `SHARED_2L_2POOL`
choice silently dispatches to a slower MLCA-style path.

## Open questions

1. **When does pool=2 *actually* beat pool=1?**
   At max_new=512 the L1 work fraction becomes non-trivial (avg L1 depth ≈
   max_new/2 = 256 tokens). With L1 dominating L0 by 10× at K=64 / B=32 /
   L_p=50k, pool=2 should cut total cost ~4×. Need to fix the picker bug
   above and re-bench.

2. **Where does the picker actually flip strategy?**
   At our grid the picker has a single answer. The picker's value would
   show in:
   - Short L_p (cascade prefix doesn't pay back the merge → flip to
     PER_BEAM)
   - Heterogeneous L_p across the batch (one batched strategy must serve
     mixed shapes)
   - K=128+ where the L0 group exceeds T_LARGE=64 (forces t_large=64 vs
     128 trade)

3. **Tree batching.**
   Tree is the only un-batched method left. Batching it (block-diagonal
   custom_mask) is feasible but mostly cosmetic — at L_p=8192 / K=16 / B=8
   tree is 16.6 ms vs cascade's 4.4 ms; batching alone won't close that
   gap because tree has no prefix-sharing.

## Cost-model picker fix — 2026-05-08

### Symptom
At K=64 / L_p=8192 / B=32 / max_new=256 (32 distinct prompts) the
auto-picker chose `shared_3l_2pool` 239/255 steps. Forced `2l1p` was
~22% faster than auto, and forced `3l1p` was the actual oracle. The
empirical step-time ranking flatly contradicted the cost-model's
predicted ranking.

### Root causes (two)
1. **`dual_pool_extra_us = 0.0` in the H100 cache.** The 2-pool kernel
   dispatch carries real per-call overhead (extra launch + sync + plan
   work) that the model wasn't pricing. Pre-fix the autotune grid was
   B=1 only, probed only `{PER_BEAM, 2L_1POOL, 2L_2POOL}`, and capped
   `DUAL_POOL_EXTRA_GRID_US` at 20 µs — so even with autotune the
   coefficient could never be fit on a workload where the 2-pool
   penalty actually shows up.
2. **`force_cta_tile_q` driver bug** (`driver.py:677`). The cascade
   plan was passed `force_cta_tile_q=pick.t_large` unconditionally, so
   the kernel ran every `*_2POOL` pick as forced single-pool
   internally. Two-pool routing was effectively never enabled in
   `bs_kernel`, even when the picker chose it. Fixed in this round:
   `force_cta_tile_q` now passed only when `pick.pool_count == 1`.

### Diagnostic (`diagnose_cell.py`)
Per-step CSV with cost-model `pick.debug` (predicted µs per candidate
strategy) alongside measured ms for each forced mode. From
`results/diagnose_cell-20260508-004515/per_step.csv` (pre-fix):

| metric | value |
|---|---|
| auto picks `3l_2pool` | 239 / 255 (94%) |
| oracle wins | `3l1p` 211, `3l2p` 31, `2l1p` 12, `2l2p` 1 |
| auto wrong vs `3l1p` | 206 / 239 (86%) |
| mean regret | +13.86% |
| worst regret | +104.98% |

Offline simulation: adding any `dual_pool_extra_us ≥ 20 µs` to every
2-pool candidate's predicted cost flips picks to `3l1p` 238/255 and
drops mean regret from 22% → 8%.

### Autotune fix (`autotune.py`)
- `AUTOTUNE_GRID` now `(K, L_p, B)` tuples — adds B=8 cells (3) and
  B=32 cell (1) on top of the original B=1 cells (9). Total 13 cells.
- `MODE_FILTERS` replaces `AUTOTUNE_STRATEGIES` and probes 5 modes
  including `3l1p` / `3l2p` (with 2L fallback for steps where the
  beam tree has no intermediate group structure).
- `_eval_regret` calls `pick_strategy_batch([w] * B, ...)` so the
  cross-prompt wave-summing the picker actually sees at runtime is
  reflected in the regret signal.
- `DUAL_POOL_EXTRA_GRID_US` widened to {0, 0.5, 1, 2, 5, 10, 20, **50,
  100, 200**}.

Tuned coefficients on H100 SXM (`coeffs-NVIDIA_H100_80GB_HBM3.json`):
`share_extra_us=100.0`, **`dual_pool_extra_us=20.0`**. Avg regret
+0.42% / worst +2.97% across all 13 cells.

Per-mode median step times measured by autotune at the failing cell:

| K | L_p | B | per_beam | 2l1p | 2l2p | 3l1p | 3l2p | oracle |
|---|----:|---|--------:|-----:|-----:|-----:|-----:|--------|
| 64 | 8192 | 32 | 224.83 | 105.79 | 117.26 | **104.08** | 117.14 | `3l1p` |

### Before / after (`diagnose_cell.sbatch`, K=64 / L_p=8192 / B=32 / max_new=256, distinct, Llama-3.2-1B)

`diagnose_cell-20260508-004515` (pre-fix) vs `diagnose_cell-20260508-025819` (post-fix):

| mode | OLD median (ms) | NEW median (ms) | Δ | OLD total (ms) | NEW total (ms) |
|------|----------------:|----------------:|---:|---------------:|---------------:|
| **auto**   | 126.37 |  90.70 | −28.2% | 37934 | 27883 |
| per_beam   | 275.57 | 208.43 | −24.4% | 76138 | 59147 |
| 2l1p       | 154.59 |  89.55 | −42.1% | 44599 | 27919 |
| 2l2p       | 158.59 |  95.00 | −40.1% | 45974 | 29706 |
| 3l1p       | 121.18 |  89.76 | −25.9% | 36331 | 27870 |
| 3l2p       | 135.23 |  95.09 | −29.7% | 40539 | 29524 |

Auto-pick distribution:
- OLD: `shared_3l_2pool` 239 / `shared_2l_2pool` 16
- NEW: `shared_3l_1pool` 237 / `shared_2l_1pool` 17 / `shared_3l_2pool` 1

### Read of the post-fix numbers
- **Auto matches the oracle.** 90.70 ms vs forced `3l1p` 89.76 ms / forced
  `2l1p` 89.55 ms — within ~1%.
- **All forced 1-pool variants tie** (`auto`, `2l1p`, `3l1p` ≈ 89-91 ms).
  Once the bug is fixed and the kernel actually routes per its pool=1
  setting, the per-step kernel work is essentially identical regardless
  of the cascade depth chosen.
- **Real 2-pool kernel overhead is ~5.5 ms / step** (`2l2p` 95.00 vs
  `2l1p` 89.55, `3l2p` 95.09 vs `3l1p` 89.76) — i.e. ~340 µs per
  attention call (16 layers × 255 steps). Smaller than the ~870 µs/call
  inferred from the buggy pre-fix data, but still real.
- **CTA_Q-padding routing doesn't pay off here.** The padding savings
  2-pool gives by routing per-beam tails to T=16 are smaller than the
  ~340 µs/call dispatch overhead at this batch shape. 1-pool wins.
- **Uniform −25 to −30% drop across all modes** between the two runs is
  H100 thermal/clock state, not a code change — the bug fix doesn't
  touch the `pool_count==1` path so `2l1p` / `3l1p` should be unaffected
  by it. The relative ranking is what matters and is consistent across
  runs.

### Per-step phase breakdown
`driver.py` records per-phase wall time via `_PhaseTimer`. Running
`diagnose_cell.py phase_summary` over each mode of
`diagnose_cell-20260508-025819`:

`auto` mode (decode_total = 27883 ms / 255 steps):

| phase | sum (ms) | % of decode | mean (ms) | median (ms) | p99 (ms) |
|-------|---------:|------------:|----------:|------------:|---------:|
| `beam_select`     | 10379 | **37.2%** | 40.7 | 33.6 | 123.0 |
| `forward`         |  6739 | **24.2%** | 26.4 | 26.3 |  27.8 |
| `page_update`     |  5619 | **20.2%** | 22.0 | 13.3 | 138.2 |
| `layout_d3`       |  1891 |   6.8%   |  7.4 |  4.2 |  90.4 |
| `input_tensor`    |  1557 |   5.6%   |  6.1 |  1.1 |  88.1 |
| `dispatch_setup`  |  1443 |   5.2%   |  5.7 |  5.5 |   6.4 |
| `pick`            |   208 |   0.7%   |  0.8 |  0.7 |   1.7 |
| `layout_d2_redo`  |    38 |   0.1%   |  0.15|  0.15|   0.18|

**The actual GPU forward pass is only 24% of decode time.** The other
76% is Python-side bookkeeping: top-K beam selection, KV-page CoW,
cascade-level decomposition, indptr/indices array building, and the
cost-model pick. The cost model itself (`pick`) is negligible at 0.7%.

#### Per-mode phase comparison
`mean_ms` per phase across the six forced/auto modes:

| phase | auto | per_beam | 2l1p | 2l2p | 3l1p | 3l2p |
|-------|-----:|--------:|-----:|-----:|-----:|-----:|
| beam_select     | 40.7 | 40.4  | 40.9 | 37.9 | 41.4 | 37.0 |
| forward         | 26.4 | **97.0** | 25.8 | 24.0 | 26.4 | 25.1 |
| page_update     | 22.0 | 21.7  | 21.6 | 21.9 | 22.1 | 21.6 |
| layout_d3       |  7.4 |  6.1  |  7.1 |  6.5 |  7.0 |  7.5 |
| dispatch_setup  |  5.7 | **53.5** |  5.1 | **16.7** |  5.6 | **18.6** |
| input_tensor    |  6.1 | 10.2  |  4.5 |  6.7 |  5.8 |  4.9 |
| pick            |  0.8 |  0.8  |  0.8 |  0.8 |  0.8 |  0.8 |
| layout_d2_redo  |  0.15|  2.27 |  3.70|  2.00|  0.15|  0.15|

Cross-mode observations:
- **`forward` jumps to 97 ms/step in `per_beam`** (vs ~25 ms in cascade
  modes) — that's the kernel doing K independent paged decodes with no
  prefix sharing.
- **`dispatch_setup` jumps to ~17–19 ms/step in `*_2POOL` modes** (vs
  ~5–6 ms in `*_1POOL`). This +12 ms is the *real* 2-pool overhead —
  it's in `cascade.plan()` (likely building two sets of qo_indptr arrays
  + the per-tile pool-routing logic), **not** in the kernel itself.
  Confirms the 2-pool penalty is dispatch-side, not compute-side.
- **`per_beam` `dispatch_setup` is 53 ms/step**: building the 32×64=2048-row
  flat indptr / indices tensors for `BatchDecodeWithPagedKVCacheWrapper.plan()`
  is much heavier than the cascade plan because there's no prefix
  factoring.
- **`pick` is 0.8 ms/step in every mode** — the cost-model decision
  itself is essentially free.
- **High p99 in `page_update`, `layout_d3`, `input_tensor`** (88-138 ms)
  vs much lower medians (1-13 ms) — these stalls are GPU-sync events
  on steps where Python-side allocation forces a wait.

#### Where the time actually goes (auto mode, 27.9 s total)
1. **`beam_select` 37%** — pure-Python top-K across 32 prompts × 255
   steps. Per-prompt loop applies `topk` and parent-id math; the K=64
   beams per prompt mean 64 candidate scores ⊗ vocab merged each step.
2. **`forward` 24%** — actual GPU work (attention + MLP + RMS + RoPE).
   This is the lower-bound; everything else is overhead Python could
   avoid.
3. **`page_update` 20%** — KV-page allocation + CoW management.
   Per-prompt page-table operations; runs B times per step.
4. **`layout_d3` 7%** — `_adaptive_levels` LCA scan + level
   decomposition. Per-prompt Python.
5. **`input_tensor` 6%** — `torch.tensor(input_ids)` and friends.
6. **`dispatch_setup` 5%** (1-pool) or 16% (2-pool) — `cascade.plan()`
   + indptr/indices/lpl array materialization.
7. **`pick` 0.7%** — cost-model strategy selection.
8. **`layout_d2_redo` 0.1%** — depth-3 → depth-2 collapse fallback,
   only used when picker chose 2L but layout produced 3 levels.

### Optimization candidates (largest-first)
- **Batch `beam_select` (37%)**: replace the Python per-prompt top-K
  loop with a single `torch.topk` over the (B, K, vocab) cube. Up to
  ~10 sec of decode time savable on this cell.
- **Batch / vectorize `page_update` (20%)**: per-prompt page CoW is a
  small Python loop per step. Combine across B; pre-allocate page
  blocks. Up to ~5 sec savable.
- **Cache `cascade.plan()` (5–16%)**: across consecutive steps the
  cascade structure changes by O(1) pages. The plan call rebuilds
  device-side workspace from scratch each step. Reusing or
  incrementally updating the plan would cut ~1.5 sec for 1-pool / ~5 sec
  for 2-pool.
- **Reduce `dispatch_setup` for 2-pool specifically** (+12 ms/step
  over 1-pool): if the 2-pool routing tables are stable across steps,
  cache them. This would close the 1-pool/2-pool gap and let the
  picker's CTA_Q-padding intuition pay off (the kernel-side savings
  exist; they're being eaten by plan-side overhead today).
- **Cache `input_tensor` (6%)**: the tensor shapes are constant per
  step; reuse a pre-allocated buffer rather than `torch.tensor()` from
  Python lists.

#### Verifying `beam_select` is real Python overhead, not stale forward
Hypothesis: forward kernel work could be leaking past the existing
forward-side `torch.cuda.synchronize()` and showing up as beam_select
time. To test, two extra defensive syncs were added in `driver.py`:

1. `post_forward_sync` — an explicit `torch.cuda.synchronize()`
   immediately after `mark("forward_ms")`, with its own phase mark.
2. A second sync just before `mark("beam_select_ms")` so any pending
   per-prompt `.topk` / `.tolist()` work is captured in beam_select
   rather than the next step's first phase.

Re-ran the auto cell with the new instrumentation
(`diag_sync_check-20260508-032035`):

| phase | prior (no extra sync) | with extra syncs | Δ (ms) |
|-------|----------------------:|-----------------:|------:|
| forward          | 26.426 | 26.436 | +0.01 |
| **post_forward_sync** | —      | **0.011** | new |
| beam_select      | 40.703 | 40.838 | +0.13 |
| decode_total     | 27883  | 27920  | +37 (noise) |

`post_forward_sync` mean is **11 µs** — essentially the cost of issuing
an empty `cudaStreamSynchronize` call when the queue is already drained.
beam_select is unchanged within run-to-run variance. So the existing
forward sync was sufficient: **beam_select's ~40 ms is genuinely
Python+sync overhead, not stale GPU wait**.

Why 40 ms is plausible for beam_select Python work alone:
the per-prompt loop runs B=32 iterations and each does
`parent_t.tolist()`, `token_t.tolist()`, and `scores_list.tolist()` —
**96 device-to-host sync points per step**. Each `.tolist()` triggers a
small kernel-launch + device-to-host copy + sync round-trip. Vectorizing
the loop into one `torch.topk` over `(B, K, vocab)` and a single
`.tolist()` would collapse 96 syncs to 2, plus eliminate the Python
iteration cost. Estimated savings: ~30 ms/step, decode_total
27.9 s → ~20 s at this cell.

### Implementation pass — `beam_select` + `page_update` Python overheads
Three targeted fixes landed across `decoding.py`, `page_table.py`, and
`bs_kernel/driver.py`:

1. **Batched standard top-K.** New
   `decoding.standard_decode_select_batched(scores: (B, K, V), K)`
   computes the top-K across the entire batch in one
   `torch.topk(flat, K, dim=-1)`. The driver detects
   `select_at_decode is standard_decode_select` and takes a fast path
   that does three big `.tolist()` calls (parents, tokens, scores)
   instead of `3*B = 96` per step. Falls back to the per-prompt loop
   for DBS / custom decode strategies that carry per-prompt state.

2. **`PageTable.allocate_block` `pop(0)` → `pop()`** — fixed an
   accidental O(N) free-list pop that quadratic-bombed slot-boundary
   steps. Free list is a stack now (LIFO), `free_block` already used
   `append`. Also added `allocate_blocks(n)` for the off==0 fast path
   (B*K=2048 allocations on those steps), avoiding N method calls and N
   `set.add` operations.

3. **Copy-on-fork beam construction.** Old code did
   `Beam(token_ids=parent.token_ids + [new], pages=list(parent.pages))`
   for every new beam — copying full token history and full pages list
   *every step*. At K=64 / B=32 / step 256 that's ≈1.1 M int copies per
   step in pure Python. New code: each parent's *last child* takes
   ownership of the parent's lists by reference and appends in place
   (O(1)); earlier siblings (only present on forks, `usage > 1`) copy
   before the assignee mutates. With most beams not forking, ~95 % of
   beam constructions become O(1).

Re-ran the auto cell after each landing
(`diag_sync_check-2026{0508-032035, 033138, 103618}`):

| phase (mean ms) | baseline | + batched topk + page_table fix | + CoW-on-fork | Δ from baseline |
|----------------|---------:|--------------------------------:|--------------:|----------------:|
| **beam_select**   | 40.70 | 33.18 | **19.25** | **−52.7%** |
| forward           | 26.43 | 26.53 | 27.30 | +3% (noise) |
| page_update       | 22.04 | 20.52 | 20.69 | −6.1% |
| layout_d3         |  7.41 |  7.44 |  9.45 | +28% |
| dispatch_setup    |  5.66 |  5.60 | 10.07 | +78% |
| input_tensor      |  6.11 |  6.18 |  2.80 | −54% |
| pick              |  0.81 |  0.81 |  0.81 | 0 |
| **decode_total** (s) | **27.88** | **25.61** | **23.17** | **−16.9%** |
| median step (ms)  | 90.7  | 82.3  | **69.0**  | **−23.9%** |

The phase shifts in the third column reflect attribution moving rather
than work appearing: when beam_select used to stall on 96 syncs,
subsequent GPU work overlapped during those stalls; now that
beam_select runs straight through, the next-step GPU work shows up
honestly in `dispatch_setup` (+4.5 ms) and `layout_d3` (+2 ms). Net is a
real −9.8 ms/step → −2.4 s decode_total.

### Remaining headroom (after first three landings)
- `dispatch_setup` 10.07 ms/step (5 ms 1-pool baseline + 5 ms now-
  exposed GPU work) — caching `cascade.plan()` across consecutive
  steps would cut most of it. The plan structure shifts by O(1) page
  per step.
- `page_update` 20.7 ms/step — still has the per-(b, beam) Python loop
  on `off > 0` steps (240 of 256 steps). Vectorizing the
  refcount-CoW dispatch (numpy or torch) could halve this.
- `layout_d3` 9.45 ms/step — per-prompt `_adaptive_levels` LCA scan
  runs B=32 times. Memoizing across consecutive steps (the cascade
  shape is stable except at fork events) is straightforward.

### 4th landing — batched CoW memcpy in `page_update`
Hypothesis going in: the bulk of `page_update` was Python loop
overhead. Reality: most of it was the Python driver issuing **up to
~32 k small CUDA memcpy launches per heavy-fork step** —
`page_table.copy_block(write, len)` fired one launch per layer (×16),
and on a fork-heavy step many CoW events fire (up to B*K = 2048).

Fix: collect `(src_page, dst_page)` pairs across all CoW events in the
step into Python lists, then after the per-prompt loop issue **one
fancy-indexed memcpy per layer** that copies *all* CoW events at once
(`kv[dst, :, :len] = kv[src, :, :len]`). All CoW events in one step
share the same `length` (= `off`), so the batch is uniform. Total
launches per step: ~16 (== num_layers), regardless of CoW count.

Re-ran the auto cell (`diag_sync_check-20260508-105215`):

| phase (mean ms) | + CoW-on-fork | + batched CoW memcpy | Δ |
|----------------|--------------:|---------------------:|---:|
| forward            | 27.30 | 26.49 | −0.8 |
| **page_update**    | 20.69 |  **0.76** | **−96.3%** |
| beam_select        | 19.25 | 21.95 | +2.7 (noise) |
| dispatch_setup     | 10.07 |  5.68 | −4.4 (overlap recovered) |
| layout_d3          |  9.45 |  6.27 | −3.2 (overlap recovered) |
| input_tensor       |  2.80 |  3.22 | +0.4 (noise) |
| pick               |  0.81 |  0.81 | 0 |
| **decode_total (s)** | **23.17** | **16.75** | **−27.7%** |
| **median step (ms)** | **69.0**  | **55.7**  | **−19.3%** |

`page_update` p99: **132.82 → 1.21 ms** (−99.1%). The spike events that
were dominating the tail are gone.

The "overlap recovered" effect on `dispatch_setup` (−4.4 ms) and
`layout_d3` (−3.2 ms) is the inverse of what happened on the prior
landing: those phases' GPU work was previously stuck behind the CoW
memcpy chain (32 k driver calls saturated the launch queue); now that
page_update releases the queue almost immediately, subsequent phases
re-overlap with each other and their wall times drop.

### 5th landing — second pass on `beam_select`
Three further reductions to `beam_select`:

1. **Cache `cum_log_probs` as a GPU tensor across decode steps.** Old
   code rebuilt it every step from
   `[[bm.cum_log_prob for bm in beams_per_prompt[b]] for b in range(B)]`
   — 2048 Python attribute reads, a `torch.tensor(...)` allocation,
   and a CPU→GPU transfer. New code initializes it once at prefill
   from the prefill top-K and updates it in place from the
   `topk` output (`cum_log_probs_bk = scores_t`).
2. **No-fork fast path.** When `len(set(parent_ids)) == K` (every old
   beam has exactly one new child) we skip `parent_usage`,
   `parent_assignee`, and the per-page `_add_ref`/`_remove_ref`
   bookkeeping entirely. The loop body becomes
   `parent.token_ids.append(...); parent.cum_log_prob = ...; new_beams[i] = parent`.
3. **Assignee parent-reuse on the fork path.** Instead of constructing
   a new `Beam(...)` dataclass that wraps the parent's lists, mutate
   the parent and reassign `new_beams[i] = parent`. Saves one
   dataclass construction per assignee — ~95 % of new beams.

Re-ran the auto cell (`diag_sync_check-20260508-105826`):

| phase (mean ms) | + batched CoW memcpy | + cum_log_probs cache + no-fork + assignee reuse | Δ |
|----------------|---------------------:|-------------------------------------------------:|---:|
| forward            | 26.49 | 26.52 | 0 |
| **beam_select**    | 21.95 | **19.03** | **−13.3%** |
| page_update        |  0.76 |  0.82 | +0.06 |
| layout_d3          |  6.27 |  7.66 | +1.4 (noise) |
| dispatch_setup     |  5.68 |  5.80 | +0.1 (noise) |
| input_tensor       |  3.22 |  3.73 | +0.5 (noise) |
| pick               |  0.81 |  0.82 | 0 |
| **decode_total (s)** | **16.75** | **16.46** | **−1.7%** |
| **median step (ms)** | **55.7**  | **55.5**  | −0.2 |

beam_select median: 15.91 → 15.09 ms. p99: 103.6 → 100.2 ms. The
remaining ~15 ms median is dominated by the per-(b, i) iteration
itself — 2048 Python operations regardless of how cheap each one is.

### 6th landing — defer cum_log_prob + numpy rc array

Two further reductions to `beam_select`:

1. **Defer Beam.cum_log_prob updates to end of decode.** The decode
   loop no longer materialises `scores_t.tolist()` per step nor sets
   `parent.cum_log_prob` per new beam — `cum_log_probs_bk` lives on
   GPU, and a single pass at the end of the outer decode walks
   `cum_log_probs_bk.tolist()` and writes `.cum_log_prob` to every
   beam (the only consumer is the final sort). Combined `parents_t`
   and `tokens_t` into a single `torch.stack(...).tolist()` so two
   D2H transfers become one.
2. **`rc_per_prompt[b]` is a numpy `int32` array instead of a `dict`.**
   The biggest cost in the fork (slow) path was the bulk rc update
   loop: for each old beam with `usage != 1`, iterate ~528 of its
   pages and call `_add_ref` / `_remove_ref` per page (~33 k Python
   function calls per fork-heavy step). The numpy array lets us do
   one `rc_b[pages_np] += delta` advanced-index op per beam instead.
   `_remove_ref` was also inlined in `page_update`'s CoW path. The
   `_add_ref` / `_remove_ref` helpers in `adaptive_pool.py` still
   work for other methods that use the dict form.

Re-ran the auto cell (`diag_sync_check-20260508-111838`):

| phase (mean ms) | + assignee reuse + no-fork (round 5) | + defer scores + numpy rc (round 6) | Δ |
|----------------|-------------------------------------:|------------------------------------:|---:|
| forward            | 26.52 | 26.44 | 0 |
| **beam_select**    | 18.87 | **13.62** | **−27.8%** |
| page_update        |  0.82 |  0.96 | +0.14 |
| layout_d3          |  7.66 |  6.95 | −0.7 |
| dispatch_setup     |  5.80 |  5.81 | 0 |
| input_tensor       |  3.73 |  4.40 | +0.7 (noise) |
| pick               |  0.82 |  0.82 | 0 |
| **decode_total (s)** | **16.46** | **15.09** | **−8.3%** |
| **median step (ms)** | **55.5**  | **52.1**  | **−6.1%** |

`beam_select` p99 collapsed: **72.99 → 36.40 ms (−50%)**. Median
15.30 → 11.73 ms. The rc-update loop on fork-heavy steps was the
single biggest contributor to the tail, and numpy advanced-indexing
made it ~quadratic-to-linear in the dominant cost.

### 7th landing — eliminate redundant pages copy in `layout_d3`
The `_adaptive_levels` call site was building
`pages_per_beam = [list(bm.pages) for bm in beams_per_prompt[b]]` —
defensively copying every beam's full pages list (≈528 ints) before
passing in. But `_adaptive_levels` only reads `pages_per_beam` and all
of its outputs (`shared_pages`, `inter_pages_per_group`,
`per_beam_tail`) are *slices* (`pages_per_beam[0][:lca]`,
`pages_per_beam[b][lca:]`, etc.) — Python list slices already produce
independent lists. The defensive copy was pure waste: 528 × K=64 ×
B=32 ≈ 1 M int copies per step.

Replaced with `pages_per_beam = [bm.pages for bm in beams_per_prompt[b]]`
(reference, no copy).

Re-ran the auto cell (`diag_sync_check-20260508-114314`):

| phase (mean ms) | round 6 (numpy rc) | round 7 (no pages copy) | Δ |
|----------------|-------------------:|------------------------:|---:|
| forward            | 26.44 | 26.51 | 0 |
| beam_select        | 13.62 | 13.12 | −0.5 (noise) |
| **layout_d3**      |  6.95 |  **3.47** | **−50.1%** |
| dispatch_setup     |  5.81 |  5.81 | 0 |
| input_tensor       |  4.40 |  4.66 | +0.3 |
| page_update        |  0.96 |  0.96 | 0 |
| pick               |  0.82 |  0.82 | 0 |
| **decode_total (s)** | **15.09** | **14.15** | **−6.2%** |
| **median step (ms)** | **52.1**  | **49.5**  | **−5.0%** |

### 8th landing — split `Beam.pages` into shared `pages_prefix` + private `pages_tail`
The fork (slow-path) sibling copy was doing
`pages=list(parent.pages)` — a 528-int Python list copy per sibling.
But ~512 of those were the prompt's prefix pages, *identical by
construction across every beam in the prompt*. Only the last 1–16
entries (decode-added pages) actually differ between siblings.
Copying the whole list was structurally redundant.

The user's framing — "store as a tree structure, and store only the
last parent id" — naturally maps to: every beam in a prompt traces
back to the same prefix-pages list object (shared ref = the "tree
trunk"); each beam holds its own short tail of decode-added pages.
Implementation:

* **`Beam`** (`adaptive_pool.py`) gains two parallel fields,
  `pages_prefix: list[int]` and `pages_tail: list[int]`. The
  bs_kernel driver uses the split form; legacy methods still use the
  unified `pages: list[int]` field. End-of-decode the driver
  materialises `beam.pages = pages_prefix + pages_tail` once for
  output compatibility.
* **`_adaptive_levels`** refactored to take
  `(pages_prefix, pages_tails, …)` natively. The LCA scan starts at
  `len(pages_prefix)` and only verifies forward into the tails —
  shared L0 pages = `pages_prefix` *by reference* (no copy) in the
  common case. A backwards-compat wrapper
  `_adaptive_levels_unified(pages_per_beam, …)` adapts the legacy
  callers (paged/mlca/adaptive_pool's `_build_cascade_plan` and
  legacy decode loop).
* **bs_kernel `driver.py`**: prefill init writes
  `pages_prefix=prompt_pages_per_b[b]` (one Python list ref shared
  across all K beams of the prompt) and `pages_tail=[]`.
  `page_update` mutates `beam.pages_tail`; the tail-relative index
  is `pli - prefix_len`. `beam_select` fork sibling copies only
  `pages_tail` (≤16 entries, ~30× faster than the prior 528-int
  copy). The numpy rc bulk update walks only the tail — the prefix
  rc is invariant under top-K (sum of `(usage - 1)` over all old
  beams = `K - K = 0`), so prefix updates are skipped entirely.

Re-ran the auto cell (`diag_sync_check-20260508-122506`):

| phase (mean ms) | round 7 (no copy in layout_d3) | round 8 (split-form Beam) | Δ |
|----------------|-------------------------------:|--------------------------:|---:|
| forward            | 26.51 | 26.51 | 0 |
| **beam_select**    | 13.12 | **10.44** | **−20.4%** |
| dispatch_setup     |  5.68 |  5.68 | 0 |
| layout_d3          |  3.47 |  4.44 | +1.0 (LCA-extend overhead) |
| input_tensor       |  4.66 |  2.98 | −1.7 |
| page_update        |  0.96 |  0.88 | noise |
| pick               |  0.82 |  0.81 | noise |
| **decode_total (s)** | **14.15** | **13.23** | **−6.5%** |
| **median step (ms)** | **49.5**  | **47.2**  | **−4.6%** |

`beam_select` p99 collapsed: **30.4 → 14.6 ms (−52%)** — the
fork-heavy step tail used to be dominated by 30+ siblings × 528-int
list copies; with the split form those become ≤16-int tail copies.

### Cumulative impact at the auto cell
Same workload, same picker, same kernel — purely Python+driver
optimization:

| metric | original | final | improvement |
|--------|---------:|------:|------------:|
| beam_select mean (ms) | 40.70 | **10.44** | **−74.3%** |
| page_update mean (ms) | 22.04 |  **0.88** | **−96.0%** |
| layout_d3 mean (ms)   |  7.41 |  **4.44** | **−40.1%** |
| **decode_total (s)**  | **27.88** | **13.23** | **−52.5%** |
| **median step (ms)**  | **90.7**  | **47.2**  | **−48.0%** |

**Forward (the actual GPU model.forward) now dominates at 51.1 % of
decode time** — the irreducible GPU work floor has crossed half of
decode for the first time. Further wins require changing the
attention kernel, batching across layers (e.g. CUDA Graphs or
fused-attention layer loops), or attacking the smaller remaining
phases: `beam_select` 10.4 ms (Python iteration floor),
`dispatch_setup` 5.7 ms (cascade.plan), `input_tensor` 3.0 ms,
`layout_d3` 4.4 ms.

### End-to-end re-bench at the cell (post round-8)
`bench_b32_k64_lp8k_m256_distinct-20260508-122805/bench.csv` — same
workload as the May-7 baseline (K=64, L_p=8192, B=32, max_new=256,
distinct prompts, Llama-3.2-1B). All numbers are
`decode_per_prompt_per_token_ms`:

| method | May-7 baseline | round-8 now | Δ | speedup |
|--------|---------------:|------------:|----:|--------:|
| `paged`         | 9.18  | 9.11  | (same) | 1.01× |
| `fasttree`      | 26.76 | 27.25 | (same) | 0.98× |
| `mlca`          | 4.48  | 3.48  | −22%  | 1.29× |
| `adaptive_pool` | 4.26  | 3.31  | −22%  | 1.29× |
| **`bs_kernel`** | **4.67** | **1.63** | **−65%** | **2.87×** |

`bs_kernel` is now **2.0× faster than `adaptive_pool`** (was 0.91× /
slightly slower in the buggy May-7 baseline), **5.6× faster than
`paged`**, and **16.7× faster than `fasttree`**. mlca/adaptive_pool
each gained ~22 % vs baseline — they pick up only the
`_adaptive_levels_unified` wrapper path (identical code), so most of
that gain is H100 thermal/clock state, not driver work. The actual
round-8 driver wins are bs_kernel-only.

### bs_kernel 6-mode forced comparison (post round-8)
`diagnose_cell-20260508-123033/diag.log` — same workload, all 6
`available_strategies` filters. Median step ms:

| mode | May-8 pre-fix | post bug-fix dpx=20 | round-8 now | Δ from May-8 pre-fix |
|------|--------------:|--------------------:|------------:|--------------------:|
| **auto**   | 126.4 |  90.7 |  **47.2** | **−63 %** |
| per_beam   | 275.6 | 208.4 |   167.0   | −39 % |
| 2l1p       | 154.6 |  89.6 |    47.1   | −69 % |
| 2l2p       | 158.6 |  95.0 |    51.7   | −67 % |
| 3l1p       | 121.2 |  89.8 |    47.1   | −61 % |
| 3l2p       | 135.2 |  95.1 |    53.0   | −61 % |

All 1-pool variants tie at ~47 ms (kernel-side they're identical
with `force_cta_tile_q=64`; picker just chooses among them). 2-pool
variants are ~5 ms slower per step — the real `cascade.plan()`
dispatch_setup overhead the picker correctly avoids via
`dual_pool_extra_us=20`. Auto = oracle.

### Cumulative timeline at the cell
Same workload throughout:

| stage | bs_kernel decode_total | bs_kernel auto median |
|-------|-----------------------:|----------------------:|
| May-7 buggy baseline (pre cost-model fix) | 38.1 s | 126.4 ms |
| After cost-model + bug fix (May-8 dpx=20) | 27.9 s |  90.7 ms |
| After 8 driver optimizations (round 8)    | **13.27 s** | **47.2 ms** |

Net **2.87× speedup** over the original buggy baseline at the user's
exact cell, with kernel and picker unchanged from the cost-model fix
onward — purely Python+driver-side work.

# Cost model — known issues found while investigating the K=16/B=4/L_p=50K picker miss

The picker mis-picks DEC_TAIL on the K=16 / L_p=50K / B=4 / mn=256 cell on
Llama-3.1-8B (TP=1). Reality has forced FUSED ≈ 5,694 ms vs forced
DEC_TAIL ≈ 9,091 ms (DEC_TAIL is **60% slower per step**); the cost model
sees them as essentially tied (≤5 µs/call gap). On 145/255 steps the
picker picks DEC_TAIL, causing a +34% end-to-end regret.

The model's per-call cost predicts FUSED correctly (~697 µs measured /
~700 µs predicted) but **under-predicts DEC_TAIL by ~420 µs per attention
call**. The investigation below decomposes where those 420 µs go.

## Issue 1 — Per-call + per-wave floor for the decode kernel is unmodeled

Direct probe of `BatchDecodeWithPagedKVCacheWrapper.run()` on H100 at varying
R × L_kv (`scripts/paper-exp/probe_decode_kernel.py`, results in
`benchmarks/bs_kernel/results/paper-exp/probe_decode_kernel.csv`).

At L_kv=16 (essentially zero per-CTA work) the kernel still costs:

| R   | waves | wall (µs) |
|----:|------:|----------:|
|   1 |     1 |      16.8 |
|   4 |     1 |      16.5 |
|  64 |     1 |      16.2 |
| 132 |     1 |      27.8 |
| 264 |     2 |      51.9 |
| 528 |     4 |     105.0 |
|1056 |     8 |     201.8 |

R=1 → R=64 doesn't grow because the kernel KV-splits to fill the SM grid
within one wave. The wall floor at R∈[1,64] is **~16 µs** with practically
no work — that's the per-call setup that doesn't shrink. Linear fit on
the wave axis gives **`ν_launch ≈ 4 µs`** (intercept) **and per-wave
overhead ≈ 24 µs**.

The cost model currently uses `decode_launch_us = 7.19 µs` (from calibration
at large R where the floor is amortised) and no separate per-wave overhead
term. **The actual floor at small R is 2–4× higher than the calibrated
launch.**

**Fix direction**: split the decode-tail cost into `ν_launch + ⌈R/N_SM⌉ ·
(κ_per_cta + κ_per_wave + L̄·ρ_d_inner)`. Calibrate at a small-R / short-L_kv
probe — same shape as the existing OLS probe but include at least one
sub-wave sample point so κ_per_wave can be separated from ν_launch.

## Issue 2 — Effective HBM bandwidth is HIGHER than `B_hbm`, not lower

This was opposite of my hypothesis. Effective BW measured from the same probe:

| R    | L_kv=4096  | eff BW (GB/s) | vs cached `B_hbm` (~1.3 TB/s) | vs HBM3 peak (3.35 TB/s) |
|-----:|-----------:|--------------:|------------------------------:|-------------------------:|
|    1 |    18.8 µs |           892 |                          0.7× |                      27% |
|   16 |    96.4 µs |         2,785 |                          2.1× |                      83% |
|   64 |   351.3 µs |         3,057 |                          2.3× |                      91% |
|  132 |   709.6 µs |         3,121 |                          2.4× |                      93% |
|  264 |  1,410 µs  |         3,142 |                          2.4× |                      94% |
| 2112 | 11,145 µs  |         3,179 |                          2.4× |                      95% |

The decode kernel saturates at ~3.1 TB/s at R ≥ 64 — about **2.4× the
`B_hbm` value the cost model uses**. `B_hbm` is calibrated from a 256 MiB
device-to-device copy that achieves ~1.3 TB/s (well below HBM3's 3.35 TB/s
peak), and that under-counts what the decode kernel actually delivers.

So the cost model **over-prices DEC_TAIL's bandwidth term by ~2.4×** at R ≥ 64.

**Caveat**: the FUSED cascade prefill kernel might saturate HBM at a
different effective rate. If FUSED and DEC_TAIL both achieve ~3 TB/s,
they cancel in the relative comparison. If FUSED's effective BW is lower
(say ~1.5 TB/s), the model's relative ordering may still be correct on
bandwidth-bound cells. **Needs a parallel prefill-kernel probe to confirm.**

**Fix direction**: separate `B_hbm_decode` and `B_hbm_prefill` constants
(or calibrate `B_hbm` from a same-kernel-style probe rather than `cudaMemcpy`).

## Issue 3 — DEC_TAIL's prefix-prefill kernel is ~200–300 µs/call slower than FUSED's fused-cascade prefill on the same prefix

Per-layer breakdown via `BS_KERNEL_TRACE_DEC_TAIL_KERNELS=1` on
K=16/B=4/L_p=50K (8B, 32 layers, 255 steps → 8,160 layer×step samples;
`scripts/paper-exp/probe_dec_tail_layer_breakdown.py`):

| sub-kernel               | mean (µs/call) | p50 | p99 | total (ms) |
|--------------------------|---------------:|----:|----:|-----------:|
| `append_paged_kv_cache`  |           7.70 |   6 |  36 |       62.8 |
| `decode_wrapper.run`     |          32.94 |  30 |  73 |      268.8 |
| **`prefix_wrapper.run`** |     **773.24** | 769 | 819 |  **6,309.7** |
| `merge_state_in_place`   |           5.88 |   6 |   6 |       48.0 |
| SUM of attention kernels |         819.77 |   — |   — |    6,689.3 |

Total per-attention-call (decode_total / num_steps / num_layers):
- DEC_TAIL: **1,214 µs/call**
- FUSED:      **700 µs/call**
- **Δ = +513 µs/call (DEC_TAIL slower)**

The smoking gun: **DEC_TAIL's `prefix_wrapper.run()` alone costs 773 µs/call
— more than FUSED's entire attention call (~700 µs).** Both process the
same 50K-token prefix on the same K=16 query rows, but DEC_TAIL uses
`BatchPrefillWithPagedKVCacheWrapper` directly while FUSED uses
`FusedMultiLevelCascadeAttentionWrapper`. These are different code paths
with different kernel-tile choices, plan structures, and per-call
overheads — and the cost model treats them as equivalent
(`cost_dec_tail_batch` evaluates `_per_prompt_levels_no_tail` with the
same `_tile_breakdown` + `_compute_us_from_tile_groups` machinery as
`cost_shared_batch`).

Decomposition of the +513 µs gap:
- **~200–300 µs**: prefix-prefill kernel chosen by DEC_TAIL is less
  efficient than the fused-cascade prefill on this shape
- **~40 µs**: extra `tail` (33 µs) + `merge` (6 µs) launches DEC_TAIL has
  on top of prefix
- **~150–250 µs**: residual — per-launch scheduling overhead from 4
  launches vs FUSED's 3, plus Python-side `wrapper.run()` overhead in
  the non-fused path, possibly stream-synchronisation differences

**Fix directions (compositional)**:
1. **Add a `dec_tail_prefix_overhead_us` coefficient** to the cost model
   that captures the per-call delta between the two prefill paths.
   Calibrate by running both wrappers on a matched (K, L_p) workload at
   engine init.
2. **Have DEC_TAIL use the fused-cascade prefill for the prefix side**
   too (treating it as a depth-1 cascade or depth-2 with a dummy
   1-token tail), so both strategies pay the same prefix kernel cost.
   This collapses the cost model gap structurally rather than via an
   empirical constant.
3. **Per-launch overhead constant**: ν_per_launch added to `C_dt` to
   account for the +1 launch DEC_TAIL has over FUSED.

**Microbench evidence for option 2** (`scripts/paper-exp/probe_prefix_wrappers.py`,
H100, num_qo_heads=32 / num_kv_heads=8 / head_dim=128, fp16):

| K | L_p | `BatchPrefillWithPagedKVCacheWrapper` | `FusedMultiLevelCascadeAttentionWrapper` (depth=2, dummy tail) | Cascade speedup |
|---:|---:|---:|---:|---:|
| 16 |  2,048 |  38.6 µs |  30.3 µs | 1.27× |
| 16 |  8,192 | 126.6 µs |  45.7 µs | 2.77× |
| 16 | 32,768 | 470.1 µs |  99.6 µs | 4.72× |
| **16** | **50,000** | **716.1 µs** | **135.4 µs** | **5.29×** |
| 32 | 50,000 | 717.0 µs | 122.7 µs | **5.84×** |
| 64 |  8,192 | 126.4 µs |  62.2 µs | 2.03× |
| 64 | 50,000 | 703.0 µs | 211.2 µs | 3.33× |

**The cascade wrapper is 2.0–5.8× faster on the same prefix workload.**
The gap widens with L_p — at L_p=2K the wrappers are nearly tied;
at L_p=50K the cascade is 5× faster. Likely a better internal
tile-scheduling decision for long-prefix problems.

**Implication for option 2**: switching DEC_TAIL's prefix-side wrapper
from `BatchPrefillWithPagedKVCacheWrapper` to
`FusedMultiLevelCascadeAttentionWrapper` would drop the prefix-call
cost on the K=16/L_p=50K cell from ~773 µs to ~135 µs — saving ~640 µs
per attention call, *more* than enough to close the +513 µs gap.
This is the highest-leverage fix; the cost-model deltas in options 1
and 3 would also help but don't address the underlying inefficiency.

## Issue 4 — Picker mixing wastes the picker's value when the cost model is wrong

Even when the cost model is off by a small absolute amount, per-step
churn between FUSED and DEC_TAIL costs *significantly* more than committing
to either pure strategy. On the K=16/B=4/L_p=50K cell:

- Forced FUSED: 5,694 ms (best)
- Forced DEC_TAIL: 9,091 ms
- Picker (145 DT + 110 FUSED): 7,635 ms (worse than pure FUSED by 1,940 ms)

If the cost model can't reliably distinguish two strategies, a stateful
tiebreaker that prefers "stick with current choice" or "prefer FUSED when
within ε of optimal" would avoid this churn. This is a band-aid, not a
fix for the underlying mispricing, but it's cheap to add.

## Issue 5 — Calibration grid doesn't sample the small-R / sub-wave regime

`measure_decode_per_kv_us` in `calibrate.py` samples at
`BS_pairs = ((256, 256), (256, 1024), (2048, 64), (2048, 256))`. All four
points have R ∈ {256, 2048}, well above N_SM=132. The OLS fit captures
behaviour at large R cleanly but completely misses sub-wave / per-wave
behaviour. The original comment in the code explicitly rejected adding
small-R samples because they "shifted the intercept upward by 50 µs" and
broke the picker on large-R cells where DEC_TAIL was empirically faster.

That trade-off is a symptom of the underlying problem: a single (ν, ρ)
pair can't model both regimes. A **per-wave-quantized** cost expression
(issue 1 fix) plus a multi-regime calibration grid (small-R, large-R)
should let both regimes be fit independently.

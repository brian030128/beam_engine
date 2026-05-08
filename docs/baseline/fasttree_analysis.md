# FastTree as a baseline — weaknesses and our novelty

Workload: `K=64`, `L_p=8192`, `B=32`, `max_new=256`, Llama-3.2-1B on H100.
255 decode steps. Per-token measured against `B*(max_new-1)`. All numbers
from `slurm/profile_fasttree_vs_bs.sbatch`.

## Final per-step breakdown

| phase    | fasttree | bs_kernel | gap   |
|----------|---------:|----------:|------:|
| forward  |  31.4 ms |   23.9 ms | 1.31× |
| **plan** | **56.6 ms** | **17.0 ms** | **3.33×** |
| topk     |   6.8 ms |    8.0 ms | 0.85× |
| cow / fork |  ~3.5 ms |  ~3.7 ms |  even |
| **per-pt token** | **3.31 ms** | **1.65 ms** | **2.0×** |

The kernel itself is competitive (1.31× slower forward). The 2.0×
per-token gap is dominated by FastTree's per-step plan cost.

## FastTree's plan, decomposed

255 steps, plan total = 56.56 ms/step:

| substep                  | ms/step | %    | category                |
|--------------------------|--------:|-----:|-------------------------|
| metadata_heuristic_loop  |  20.29  | 35.9 | **FT algo intrinsic**   |
| build_radix_tree         |  14.61  | 25.8 | data structure          |
| expand_to_slots          |   9.21  | 16.3 | data structure          |
| metadata_req_reduction   |   3.72  |  6.6 | **FT algo intrinsic**   |
| metadata_vnode_pack      |   2.75  |  4.9 | **FT algo intrinsic**   |
| metadata_tensor_build    |   2.19  |  3.9 | host↔GPU                |
| write_slots              |   0.50  |  0.9 | glue                    |
| gather_paths             |   0.06  |  0.1 | glue                    |
| **sum**                  | **53.31** | 94.3 | (3.25 ms counter overhead) |

Group totals:
- **FastTree algorithm intrinsic — 26.8 ms (47%)**: heuristic loop +
  vnode pack + req reduction. Floor.
- **Page→slot data structures — 23.8 ms (42%)**: radix tree + slot
  expansion. Required to feed FastTree's slot-indexed kernel.
- **Host↔GPU + glue — 2.7 ms (5%)**: ~optimal after batched H2D.

## What we tried, what worked, what didn't

We applied bs_kernel's caching pattern to fasttree to make the
comparison fair. Five integration optimizations, in order of impact:

| optimization                                   | plan ms/step | Δ   |
|------------------------------------------------|-------------:|----:|
| baseline (after rewrite into page_driver)     |       124.94 |   — |
| LCA-prefix-skip in radix walker                |        89.50 | -28% |
| numpy-vectorized `_expand_pages_to_slots`      |        ~75   | -7  |
| batched H2D via numpy bridge (11 tensors → 1) |       63.09  | -12 |
| (final, no instrumentation)                    |       56.56  |     |

We also tried whole-tree cache (multiset-keyed). It only achieved
**15.3% hit rate** and saved <5 ms/step net. Reason: FastTree's tree
shape depends on the per-prompt path multiset, which beam-search
dynamics shifts every step (topk fork changes parent multiplicities).

## FastTree's structural weaknesses

These are properties of the FastTree algorithm itself, exposed by
running it at long-context beam search at scale.

### W1. Plan dominates wall clock at scale

At K=64/B=32/L_p=8K, plan = 56.6 ms vs forward = 31.4 ms. **Plan is
1.8× the kernel itself.** The kernel is fast precisely because it
pushes routing decisions into the host-side planner — an explicit
cost-shifting choice that doesn't pay back when the workload is
plan-dominated. This worsens with K and B (more beams per prompt =
larger tree, more nodes for the heuristic to score).

### W2. Per-step routing decisions, no monotone state to exploit

`_tree_heuristic` runs up to 3× per step in an iterative refinement
loop. Its inputs:
- tree topology (changes when path multiset changes — every step);
- per-node `seqlen` in tokens (grows every step as we write new K/V);
- KV split size and Q tile size thresholds (autotuned offline).

There is **no monotone scalar state** that summarizes "the heuristic's
decisions are unchanged from last step." Compare to bs_kernel's LCA:
LCA only grows, so when LCA is unchanged, the entire prefix-wrapper
plan is reusable.

We attempted a topology cache keyed by per-prompt sorted path multiset
(order-invariant under topk reorder). Hit rate: **15.3%**. By contrast,
bs_kernel's LCA cache hits **~94%** — a 6× gap in cache effectiveness
that reflects the underlying state granularity.

### W3. Tree shape is multiset-keyed, not LCA-keyed

A page-level radix tree captures the *full* divergence pattern of K
beams (every shared run, every split point), not just the deepest
universally-shared depth. Page-level LCA is monotone (grows ~once
every 16 steps); the tree's branching pattern fluctuates every step
because:
- topk fork redistribution: parent A goes from 1 child → 3 children
  while parent B drops 2 → 0;
- a new tail page allocation at any beam shifts that beam's path;
- a single beam dropping out of top-K shifts the multiset.

Any of these flips the multiset → cache miss → full rebuild.

### W4. Slot-level kernel input from page-level state

FastTree's Triton kernel reads K/V via **flat slot indices**
(`slot = page * page_size + offset`). Even though the data structure
the planner reasons over is page-level (16× cheaper after our
optimization), the *kernel input* is slot-level: ~262K int32 slot
indices per step at L_p=8K/B=32, which must be materialized in
host memory and copied to GPU every step.

`expand_to_slots` (page→slot) is 9.2 ms/step in numpy-vectorized form;
in the original Python-loop form it was 13.2 ms. Cannot go to ~0 unless
the kernel is rewritten to take page-level inputs.

### W5. Cost model is open-loop

`(alpha, beta, gamma)` and `(KV_SPLIT_SIZES, para_threshs1,
para_threshs2)` are autotuned **once offline** on a calibration grid.
At inference time these constants don't adapt to:
- the actual `(B, K, L_p)` of the current workload;
- the per-step shape (e.g., the LCA depth in pages);
- runtime measurements of past steps' actual cost.

When FastTree's autotuned regime doesn't match the deployment
workload — exactly our case at K=64/B=32 long-context — the heuristic
falls back to suboptimal SplitQ/SplitK choices. There is no per-step
correction.

### W6. Two-stage attention with online softmax merge

Stage 1 emits per-(vnode, q-tile) attention outputs. Stage 2 reduces
across vnodes per request via online softmax. Each stage is a separate
Triton launch with its own metadata copies. FlashInfer's
prefill+decode pair (used by bs_kernel) is also two kernels but is
orchestrated by a single C++ scheduler call inside `wrapper.plan()`,
amortizing dispatch overhead.

## Our novelty over FastTree

bs_kernel is what we propose; FastTree is the closest published
baseline. Concrete differences:

### N1. LCA cache exploits monotone shared state

We recognize that the **single integer per prompt** that summarizes
"how much of the page path is universally shared" — the LCA depth — is
the structurally stable element across steps. LCA only grows; we cache
the prefix-wrapper plan keyed by LCA and re-plan only when LCA shifts
(typically once per 16 decode steps, when a new tail page is
allocated). **94% hit rate** vs FastTree's tree-multiset 15%.

### N2. Online-calibrated cost model

`calibrate.py` runs per-tile probes (`_T_tile_cost`,
`_decode_tile_cost`) at startup to measure the **actual cost** of each
kernel tile size on the current GPU. The picker (`pick_strategy_batch`)
combines these with the per-step `(B, K, depth, kv_len)` shape to score
each candidate strategy. No offline autotuning; no static thresholds.

### N3. Decode-tail kernel via online softmax merge

Per-beam tail attention has CTA_Q=1 logically (1 query × K beams).
FlashInfer prefill kernel tiles are {16, 64, 128} → 98.4% padding at
K=64. We dispatch this level through `BatchDecodeWithPagedKVCacheWrapper`
(purpose-built for CTA_Q=1, 0% padding) and merge with the prefix
wrapper output via `merge_state_in_place`. Padding goes 98.4% → 0%
on the bottleneck tile.

### N4. Cascade depth + pool count as runtime-tuned

bs_kernel exposes `(depth ∈ {2L, 3L}) × (pool ∈ {1POOL, 2POOL}) ×
(tail_kernel ∈ {prefill, dec_tail})` as picker dimensions. The picker
chooses per step based on the cost-model coefficients and the current
shape. FastTree has no equivalent — its kernel is monolithic.

### N5. Plan caching with two key structures

bs_kernel caches:
- **prefix_prefill plan**, keyed by `last_lca_per_prompt` (~94% hit);
- **decode_wrapper plan_info**, keyed by max KV length (re-plan only
  when split-K factor would shift; ~95% hit).

Two cheap monotone scalars cache the entire FlashInfer C++ scheduler
state. FastTree has no analogous separation between cacheable and
volatile metadata.

### N6. C++ scheduler reuse

By leveraging FlashInfer's wrapper plans, our plan-time work is mostly
a C++ scheduler call (microseconds for cache hits). FastTree's plan
is Python-level loops over tree nodes (milliseconds even at the floor).
This is not just an implementation choice — it follows from N1: the
LCA-keyed cache key is what makes plan-time C++ delegation viable.

## One-line summary for a paper

> bs_kernel formulates the beam-search attention plan as a tunable
> over (cascade depth, pool count, tail-kernel choice), exploits a
> single monotone scalar (page-level LCA) to cache the structurally
> stable portion across decode steps, and dispatches the per-beam
> tail through a CTA_Q=1 decode kernel merged with the prefix via
> online softmax — yielding 2.0× per-token speedup over FastTree at
> K=64/L_p=8K/B=32 with no precision loss. FastTree's tree-multiset
> state granularity admits at best 15% cache hit and forces per-step
> Python-side routing decisions, structurally limiting how cheap its
> plan can become.

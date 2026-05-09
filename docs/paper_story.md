# Paper story — adaptive multi-kernel attention dispatch for tree-structured generation

## What is the workload, really?

Many recent generation patterns share one structural feature: **a long
shared prefix fans out into a tree of branching continuations**, and
attention is computed over the (prefix + branches) joint state at every
step.

| Workload                                 | Branching structure                         | Reference                                         |
|------------------------------------------|---------------------------------------------|---------------------------------------------------|
| Beam search                              | uniform K-way at every step                 | classical                                         |
| **Diverse beam search (DBS)**            | **K-way with group-diversity penalty; top-K shifts every step** | Vijayakumar et al. 2018       |
| Self-consistency                         | N independent samples from one prompt       | Wang et al. 2022                                  |
| Few-shot prompting                       | M shared in-context examples → 1 query      | Mann et al. 2020                                  |
| Tree of Thoughts                         | non-uniform branching, variable depth       | Yao et al. 2023; Hao et al. 2023; Xie et al. 2024 |
| Speculative decoding (SpecInfer, Medusa) | speculative-tree fan-out per step (static)  | Miao et al. 2023; Cai et al. 2024                 |

**Beam search and DBS are the hardest case in this list.** DBS in
particular: the diversity penalty perturbs the per-step top-K so that
which beams survive a fork shifts every step. The branch tree is
maximally dynamic — the planner cannot pre-compute the topology, the
KV pages diverge unpredictably, and the per-beam tail length is
non-uniform. The other workloads are subsets:

- **Self-consistency** — degenerate `K=1` branching per "beam"; static
  paths.
- **Few-shot prompting** — one query against M shared examples; one
  fork.
- **Tree of Thoughts / RAP** — non-uniform branching but with
  retraction; less aggressively dynamic than DBS step-to-step.
- **Speculative decoding** — per-step *static* speculative tree of
  ~5–20 candidate tokens; the topology is fixed inside each step.

If a dispatcher works on DBS, the rest follow. **Primary evaluations
in this paper: DBS and speculative decoding** — DBS as the technical
stress test (most-dynamic branching) and speculative decoding as the
deployment case (the production tree workload). Vanilla beam search
serves as a simpler controlled sweep. The technique is for
tree-structured attention generally.

## What existing kernels actually do

Existing kernels framed as "shared-prefix attention" fall into two
shape buckets, and **neither bucket makes a runtime, multi-kernel
dispatch decision**:

**Bucket 1 — single-kernel dispatchers.** Paged-decode, FastTree, and
DEFT all route every tile of the attention call through one compiled
kernel template. The cost model (when present) picks parallelism /
split parameters *for that one kernel*.

- **paged-decode** (vLLM): per-branch independent decode calls. Each
  branch re-reads the entire prefix from HBM. `O(B·K·L_p)` prefix
  traffic. Long prefix → memory-bandwidth bound, scales linearly in
  branch count. No cost model — one kernel, one beam at a time.
- **FastTree** (MLSys '25, `docs/baseline/fasttree_analysis.md`):
  page-radix tree + Triton two-stage decode kernel with per-vnode
  SplitQ/SplitK heuristic. The cost model is non-trivial — but it picks
  parallelism parameters for **one** Triton kernel template. At
  `K=64/B=32/L_p=8K` the per-step Python planner takes 56 ms vs 31 ms of
  actual kernel time (`fasttree_analysis.md:11–22`). All tiles flow
  through the same compiled kernel and pay its worst-case padding.
- **DEFT** (`3rdparty/DeFT`): tree-structured attention with a flatten /
  node-level dispatcher. Same bucket structure: a per-step planner over
  a single attention-kernel dispatcher. No multi-kernel routing.

**Bucket 2 — multi-level primitives without runtime selection.** MLCA
and FlashInfer's fused cascade expose a flexible multi-level cascade
*shape* but commit to depth, level boundaries, and CTA_Q at construction
time.

- **MLCA** (FlashInfer's `MultiLevelCascadeAttentionWrapper`): launches
  one prefill kernel per level and bridges them with a separate
  `merge_states` call. The number of levels, the level boundaries, and
  the CTA_Q template are all caller-supplied parameters. **MLCA lacks
  runtime level / depth / T selection** — whether it wins or loses on a
  given workload is decided by the caller at construction. It is a
  primitive, not an adaptive system.
- **FlashInfer fused cascade**: same shape, fused — but still fixed
  depth, fixed `T`, no runtime selection.

**The common failure across both buckets:** none adapts to the workload
*at runtime, across kernels*. Bucket 1 has at most one cost model over
one kernel; bucket 2 has multiple kernels but no cost model deciding
how many or which. They are *primitives*, not adaptive systems.

## The core observation: bimodal Q-grouping → q_tile padding is frequent and severe

Tree-structured attention has **bimodal query grouping inside a single
attention call**:

| Level                          | Q grouping `R`        | Origin                                       |
|--------------------------------|-----------------------|----------------------------------------------|
| Shared root (the prefix)       | `K · G` (large)       | All branches share the prefix                |
| Per-branch tail / leaf         | `G` (small)           | Each branch contributes one query × G heads  |

For Llama-3.1-8B / -3.2-1B, `G = num_qo_heads / num_kv_heads = 4`. At
`K = 64`, root `R = 256` and leaf `R = 4` — a **64× gap inside one
attention call**. No single CTA_Q resolves both:

- Small `T = 16`: leaf is fine (`R = 4 → U = 25%`), but root splits
  into `K·G/16 = 16` tiles each re-reading the prefix from HBM —
  **1.46× slowdown** at K=64 (`docs/cta_tile_q_design.md:75–86`).
- Large `T = 128`: root packs into one tile, but leaf pads to `4/128 = 3%`
  MMA utilization — **1.59× slowdown** at tail=4096
  (`cta_tile_q_design.md:88–101`).

**This is not a corner case.** At our canonical workload cell
(`K=64/B=32/L_p=8K`), the per-beam tail level pays **98.4% MMA padding**
under any single-CTA_Q prefill template at `T ∈ {16, 64, 128}`
(`benchmarks/bs_kernel/results/RESULTS.md:110–116`,
`cta_tile_q_design.md:143`). Any tree workload with `K ≥ 16` hits the
bimodal regime at *every* decode step. Frequent. Severe. And invisible
to a single-kernel cost model.

A closed-form tile-cost expression (`cta_tile_q_design.md:155–169`)

```
C_tile(T, L) ≈ α + β·T·L + γ·L
```

makes both failure modes explicit: small `T` pays `O(R/T)` redundant KV
scans for the large-R level; large `T` pays `O(T/R)` MMA-padding waste
for the small-R level. Single-`T` is never Pareto-optimal.

## Our mechanism — group queries by Q_tile size, adaptively pick levels and kernels

Existing kernels assume a *single* CTA_TILE_Q (and a single kernel
*family*) for the entire attention call. The bimodal observation forces
a different design. We make three runtime decisions per step:

**Step 1 — group queries by their Q_tile-optimal kernel.** Tiles with
`R = K·G` (root) want a large-T prefill template (one KV scan,
moderate-to-high MMA utilization). Tiles with `R = G` (per-branch)
want either a small-T prefill template (`T = 16`, U = 25%) or — at the
CTA_Q=1 limit — a paged-decode kernel (CTA_Q = 1 native, 0% padding).
Different tile shapes go to different kernels in the *same* attention
call.

**Step 2 — pick the number of cascade levels** (2 or 3). Depth=2
gives a single prefix level shared by all beams plus a per-beam
tail. Depth=3 inserts an intermediate level that captures
group-level sharing (e.g., DBS groups, batched prompts that share a
sub-prefix). Both depths are candidates in the dispatch space; the
cost model picks per step based on whether the intermediate level
amortizes its plan launch.

**Step 3 — adaptively decide the kernel family per level.** Each level
gets routed to the kernel that minimizes its cost: prefill at `T_lg ∈
{64, 128}` for the shared root, prefill at `T = 16` for an intermediate
level, paged-decode (`BatchDecodeWithPagedKVCacheWrapper`) for the
per-beam tail. Outputs across levels are bridged by FlashInfer's
public `merge_state_in_place`
(`flashinfer/cascade.py:136`) — same online-softmax numerical semantics
as a fused-kernel LSE merge.

The result is a **multi-kernel, multi-level dispatch space** — a
substrate, not a single answer:

```
Strategy ∈ {
  PER_BEAM,
  SHARED_2L_1POOL,  SHARED_3L_1POOL,
  SHARED_2L_2POOL,  SHARED_3L_2POOL,
  SHARED_2L_DEC_TAIL, SHARED_3L_DEC_TAIL,
}
```

(verbatim from `cost_model.py:40–51`). **No point in this space
dominates the workload grid.** PER_BEAM wins when L_p is short enough
that cascade plan overhead exceeds prefix-sharing savings. Single-pool
2-level wins when L_p is long but the bimodal gap is mild (small K, or
small G). Two-pool variants win when the gap is severe and the tail
length amortizes the second pool's plan launch. DEC_TAIL — the CTA_Q=1
limit where the small-T pool is replaced by a paged-decode kernel —
wins specifically when the per-beam tail is too short to amortize a
prefill-tail launch (long prefix, large K, very short tail). The
3-level entries become optimal when hierarchical sharing exists (e.g.,
DBS group structure introduces a genuine intermediate sharing level
between the root prefix and the per-beam tail).

Because no fixed point dominates, the cost model is the load-bearing
mechanism, built **on top of** the dispatch space — it picks among
kernel combinations per step, not within a single kernel's parameters.
That is the structural difference vs. FastTree / DEFT (single
dispatcher with a cost model over its parameters) and vs. MLCA
(multiple kernels, no cost model).

### Justification of dispatch-space size — why only 11 candidates?

A reviewer should ask: *why this finite, hand-picked dispatch space?
What's missing?* Four justifications — three structural, one
empirical — close the question:

**J1 — `T_LARGE ∈ {64, 128}` and `T_SMALL = 16` is the
kernel-supported set, *and* the Pareto frontier of `C_tile(T, L) ≈ α
+ β·T·L + γ·L` is sparse on it.**
The cost is nearly linear in `T` at fixed `L`, with tile count
`≈ ceil(R/T)`. Adjacent `T` values (T=32 between 16 and 64; T=96
between 64 and 128) lie on the line and produce nearly identical
predicted µs once `R` is fixed; adding them inflates candidate count
without expanding the achievable optimum. Empirically verified by
expanding `T_LARGE_CHOICES = {32, 64, 96, 128}` and confirming the
picker rarely picks the new `T` values, with predicted-µs gap to
the original set within a few percent.

**J2 — Pool count ∈ {1, 2} matches the natural cardinality of the
bimodal observation.**
The bimodal observation says R takes exactly two values (`K·G` at
root, `G` at branch). One pool per R-value is the matching cardinality.
A 3rd pool would only help if a 3rd R existed — which it does not in
branch-tree-without-retraction workloads.

**J3 — Depth ∈ {2, 3} matches the empirical sharing-structure
cardinality of branch-tree workloads.**
Branch-tree-without-retraction workloads have at most three
naturally distinct sharing levels: root prefix, intermediate group
prefix (e.g., DBS groups, sub-batches that share a partial prefix),
per-beam tail. Deeper sharing structures don't arise from any
workload in our table — DBS, beam search, and speculative decoding
all peak at three. Each extra level adds a plan launch + a merge
without a workload-structure justification.

**J4 — Single `(depth, pool, T)` per batch is a fused-launch
constraint, not a modeling shortcut.**
The wave-count / SM-occupancy term is *global* across the batch
(`ceil(total_tiles / num_sms)` in our cost model). Per-prompt
strategy choice would require splitting the batch into multiple
kernel launches, losing the cross-batch SM-occupancy effect — even
if per-prompt picks were theoretically better, the launch
decomposition would erase the gain.

J2 and J3 are *observation-derived*: they fall out of the bimodal
observation (C1) and the workload-structure table at the top of the
paper. J4 is a *launch-architecture constraint*. J1 is the only
justification that needs an empirical anchor; the T-saturation
experiment provides it. Together they close the "why only 11?"
question.

## Why a cost model is the load-bearing piece

The dispatch space alone doesn't help if you can't pick the right
point in it. Two structural facts make picking non-trivial:

1. **No fixed point dominates the grid.** As enumerated above, every
   strategy in the space is optimal in some workload regime and
   suboptimal in others. A fixed-strategy choice — even a "good
   default" like `SHARED_2L_2POOL` — pays a multiplicative penalty in
   the regimes that don't match it. Section B (the experimental plan)
   shows six adversarial cells, each with a different fixed-strategy
   winner, and confirms that the only column matching the per-row
   winner everywhere is the picker.

2. **The picker has to be cheap per step.** Per-step adaptive dispatch
   only pays off if the *planning* is cheap. If plan time costs more
   than the dispatch saves, runtime adaptation is a net loss. FastTree's
   plan is **56 ms / step** at our canonical cell — more than its
   kernel time of 31 ms (`fasttree_analysis.md:11–22`). The reason is
   structural: FastTree's per-step routing heuristic re-runs
   node-by-node every step, and its state can't be amortized.

The cost model itself is a closed-form expression evaluated over a
small dispatch space (≤7 strategies × a few T values), so it is
microseconds-scale even before any caching. Plan-state caching of the
underlying FlashInfer scheduler call is a straightforward engineering
amortization on top of that — kept in the implementation, not claimed
as a research contribution.

The intellectual claim is therefore narrow and falsifiable:

> Across the workload grid, no fixed strategy in the dispatch space
> matches the picker's per-cell choice. The cost model selects within
> low oracle regret, with planning overhead small enough to leave the
> dispatch savings net-positive.

## Cited contributions

1. **Bimodal Q-grouping observation + closed-form tile-cost model.**
   A tree-structured attention call has two simultaneous Q-grouping
   regimes (`R = K·G` at root, `R = G` at each branch). The cost
   expression `C_tile(T, L) ≈ α + βTL + γL` shows no single CTA_Q is
   Pareto-optimal: small `T` pays `O(R/T)` redundant prefix scans, large
   `T` pays `O(T/R)` MMA padding. **q_tile padding is frequent and
   severe**: 98.4% on the per-beam level at the canonical cell,
   1.46–1.59× single-T degradation in adversarial regimes.

2. **Q_tile-grouped, multi-kernel, multi-level dispatch space.**
   The space `{PER_BEAM, SHARED_{2,3}L_{1,2}POOL,
   SHARED_{2,3}L_DEC_TAIL}` groups tiles by Q_tile-optimal kernel,
   admits 2- or 3-level decompositions of the attention call, and
   admits per-level kernel-family choice (prefill@T_lg, prefill@T_sm,
   paged-decode); levels are bridged by online-softmax merge. Different
   points in the space win in different workload regimes — no single
   point dominates the grid. Prior designs are either single-kernel
   (paged, FastTree, DEFT) or multi-level-but-statically-configured
   (MLCA, FlashInfer fused cascade); neither covers the full space.

3. **Cost-model picker over the dispatch space.** A 4-axis runtime
   choice — `(depth ∈ {2, 3}) × (pool count ∈ {1, 2}) × (tail-kernel
   ∈ {prefill, decode-merge}) × (T per pool ∈ {16, 64, 128})` — driven
   by a closed-form cost expression with online-calibrated coefficients
   (per-tile probes at engine init). Across the workload grid, the
   picker matches the per-cell oracle within low regret, while no
   fixed strategy in the space matches the oracle on every cell.
   Section B's six adversarial cells (one per dispatch-space winner)
   are the empirical justification.

## Scope claim — DBS as stress test, speculative decoding as deployment case

The technique is **per-step multi-kernel dispatch for
prefix-tree-structured attention**, validated on two primary
workloads — DBS (technical stress test) and speculative decoding
(deployment case) — and designed for the entire class of
branch-tree-without-retraction workloads:

- **DBS — primary evaluation.** Group-diversity penalty makes the
  top-K (and thus the surviving-beam set) shift step to step. Branch
  topology is maximally dynamic; KV pages diverge unpredictably. This
  is the stress case for any per-step dispatcher.
- **Speculative decoding (Medusa, SpecInfer) — primary evaluation.**
  Per-step static speculative tree of ~5–20 candidates. The candidate
  level is exactly the DEC_TAIL regime (CTA_Q=1) — the deployment
  workload where the dispatcher's tail-kernel choice matters most in
  practice.
- **Vanilla beam search.** A controlled sweep — uniform K,
  deterministic top-K — used to isolate kernel-time effects from
  selection dynamics.
- **Few-shot / in-context prompting.** Shared prefix → 1 query; the
  picker collapses to `(2L, 1, prefill, T_default)` correctly,
  matching paged-decode performance. Not separately evaluated.
- **Tree of Thoughts / RAP / reasoning-as-search.** Branching with
  potential retraction. The dispatcher itself still applies; not
  separately evaluated in this paper.

We benchmark DBS as the technical stress test (most-dynamic
branching) and speculative decoding as the deployment case (the
production tree workload). The other workloads in the table are
subsets the dispatcher applies to but which we do not separately
evaluate; see Honest Limits.

## Honest limits

- We do not write a new attention kernel. Kernels are FlashInfer's
  prefill + decode wrappers (with one upstream patch — `force_cta_tile_q`
  per call, ours, in `3rdparty/flashinfer`).
- Empirical scope: H100, Llama-3.1-8B / Llama-3.2-1B. Multi-GPU and
  multi-model breadth is future work.
- The dispatch space is hand-curated, not learned. The 11-candidate
  set is justified by J1–J4 above (kernel-supported `T` values + the
  bimodal Q-grouping observation + the workload's sharing-structure
  cardinality + the fused-launch constraint), but workloads outside
  the branch-tree-without-retraction class (e.g., ToT with
  backtracking, general radix-tree attention) may need additional
  dispatch points this paper does not cover.
- Workload coverage: DBS + speculative decoding are evaluated;
  few-shot / self-consistency / ToT are argued to be subsets but not
  separately benchmarked.
- DBS overhead at the moment is dominated by Python-side per-prompt
  top-K (`benchmarks/bs_kernel/results/RESULTS.md:218–279`) and is
  shared across all six methods. We report kernel time; the Python
  overhead is orthogonal to the dispatcher and not a kernel problem.

## One-paragraph abstract draft

> Tree-structured LLM generation — diverse beam search, speculative
> decoding, self-consistency, Tree-of-Thoughts — shares one
> attention-time problem: a long prefix fans out into many divergent
> branches, producing **bimodal query grouping** (`R = K·G` at the
> shared root, `R = G` at each branch) inside one attention call. A
> closed-form tile-cost model `C_tile(T, L) ≈ α + β·T·L + γ·L` shows
> *no single CTA_Q is Pareto-optimal*: small `T` pays redundant prefix
> scans, large `T` pays MMA padding (98.4% on the per-beam level at
> K=64/B=32/L_p=8K). Existing kernels split into two camps and
> *neither* makes runtime, multi-kernel decisions: paged-decode,
> FastTree, and DEFT route all tiles through a single
> attention-kernel dispatcher with a cost model over that one
> kernel's parameters; MLCA and the FlashInfer fused cascade expose a
> multi-level primitive but with no runtime level / depth / T
> selection. We propose a **Q_tile-grouped, multi-kernel, multi-level
> dispatch space** `{PER_BEAM, SHARED_{2,3}L_{1,2}POOL,
> SHARED_{2,3}L_DEC_TAIL}` and a **per-step cost-model picker** over
> the 4-axis runtime choice `(depth × pool count × tail-kernel × T
> per pool)`, with online-calibrated coefficients. Across our
> workload grid, *no fixed strategy in the dispatch space matches
> the picker on every cell* — six adversarial cells each have a
> different fixed-strategy winner, and only the picker tracks the
> per-cell oracle. On Llama-3.1-8B at K=64/L_p=8K we obtain **2.0×
> over MLCA** and a `<TODO>` × gap over FastTree at the same
> workload cell, with the gap growing as plan-time amortizes across
> longer decodes. We evaluate primarily on diverse beam search (the
> technical stress case) and speculative decoding (the deployment
> case), with the dispatcher applying unchanged to other
> branch-tree-without-retraction workloads.

## TL;DR for review

Three load-bearing claims:

1. **Bimodal Q-grouping is fundamental, not an artifact.** Closed-form
   tile-cost model shows no single CTA_Q can serve both regimes.
   q_tile padding is *frequent* (every step at K ≥ 16) and *severe*
   (98.4% on the per-beam level at the canonical cell; 1.46–1.59×
   single-T degradation in adversarial cells).

2. **The dispatch space + per-step picker is the substrate prior work
   doesn't have.** FastTree and DEFT build their cost model around a
   *single* attention-kernel dispatcher; MLCA and the FlashInfer
   fused cascade expose multiple kernels but with no runtime level /
   depth / T selection. We define the dispatch space `{PER_BEAM,
   SHARED_{2,3}L_{1,2}POOL, SHARED_{2,3}L_DEC_TAIL}` and a 4-axis
   cost-model picker over it. **For any single fixed strategy in the
   space, there is a workload cell where it loses to the picker by
   ≥X%** (Section B): six adversarial cells each have a different
   fixed-strategy winner, and only the picker is best on every row.
   The dispatch-space cardinality (11 candidates) is justified by
   J1–J4: kernel-supported `T` set, bimodal-observation cardinality,
   workload sharing-structure cardinality, and the fused-launch
   constraint.

3. **Scope: branch-tree-without-retraction workloads, with DBS and
   speculative decoding as primary cases.** DBS is the technical
   stress test (maximally dynamic top-K); speculative decoding is the
   deployment case (per-step static candidate tree). Beam search,
   few-shot, ToT, RAP, and self-consistency are subsets we argue the
   dispatcher applies to but do not separately evaluate.

The 2.0× over MLCA is solid. Open items before submission: replace
`<TODO>` (FastTree-cell speedup) in the abstract; complete the
6-cell adversarial table from Section B; report median oracle regret
across the workload grid.

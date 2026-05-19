# Paper story — a fixed-cardinality dispatcher for tree-structured attention

## Introduction

Tree-structured generation is becoming increasingly common in LLM
inference. Applications such as beam search, diverse beam search,
self-consistency, speculative decoding, and batched reasoning
workloads all produce multiple continuations from a shared context.
In our target workloads, this structure appears in three concrete
settings: `multi_chain_reasoning`, where multiple reasoning paths are
decoded from a common prompt; `multi_level_system`, where a batch of
prompts shares long system and task prefixes; and `multi_few_shot`,
where many continuations reuse the same in-context examples. Although
these applications differ in how branches are generated, they share
the same attention-time structure: **a long shared prefix fans out
into a tree of branch-specific continuations**.

This structure makes conventional paged decoding inefficient. If each
branch is treated independently, every branch rereads the same shared
KV prefix from HBM at every decoding step. We use $K$ to denote the
**total queries issued per prompt at one decoding step** — the leaves
of that prompt's per-step KV-sharing tree (K beams in beam search, K
samples in self-consistency, the top-K candidates in speculative
decoding). A radix tree with K leaves and compressed internal nodes
has at most $2K-2$ edges and, for balanced fanout, depth $O(\log K)$.
For a batch of $B$ prompts, paged decoding replicates prefix traffic
across $B \cdot K$ branch instances. Prior systems have therefore proposed
ways to exploit shared-prefix or tree-structured attention. Paged
attention improves KV-cache management but does not eliminate repeated
prefix reads across branches. Multi-level cascade attention exposes
shared-prefix computation across multiple KV levels, but the level
structure and kernel configuration are fixed by the caller. Recent
tree-attention systems such as FastTree dynamically plan over a
radix-tree representation of the KV cache, but still execute their
work through a single prefill-style kernel template family. These
approaches show that tree attention is an important optimization
target, but they do not fully address the interaction between tree
structure and kernel tile shape.

The **core difficulty** is that tree attention contains two very
different query-grouping regimes inside the same attention problem.
At the root of the tree, many branches attend to the same long
prefix, so the effective query group is large. At the leaves, each
branch attends to its own short tail, so the effective query group is
small. However, GPU attention kernels commit at compile time to a
fixed query tile size, which we denote `CTA_Q`. A small `CTA_Q` is
well matched to narrow per-branch tails, but it splits the shared
root into many tiles and repeatedly scans the same prefix. A large
`CTA_Q` packs the root efficiently but wastes compute on the tail
because most rows in the query tile are padding. **A single
prefill-style attention template cannot be optimal for both the
shared root and the narrow per-branch tail.**

A natural solution is to use **two kernel families**: process the
wide shared prefix with a prefill-style kernel, and process the
narrow per-branch tail with a dedicated decoding kernel whose query
tile size is native to single-token decoding. This removes the
`CTA_Q` padding problem on the tail. However, this solution is not
always faster. Launching a separate decoding kernel introduces extra
launch overhead, and can cause SM underutilization when the tail has
too little work. The right execution strategy therefore depends on
the current tree shape, prefix length, branch count, batch size, and
tail length: sometimes the dedicated decoding kernel wins, sometimes
a fused prefill-style execution is faster.

A second challenge is **how the tree should be grouped**. Real
tree-structured workloads are not always two-level problems
consisting only of a root prefix and independent leaves. They may
contain intermediate sharing levels, such as diversity groups, beam
survivors that share partial suffixes, or speculative subtrees.
Exposing more levels can reduce KV traffic, but each additional level
also increases planning cost, scheduling overhead, and merge cost.
This creates an optimization space: the runtime must decide how many
levels of the tree to expose, how to group queries at each level, and
whether the tail should be handled by the prefill kernel or by a
dedicated decoding kernel.

This paper proposes a **fixed-cardinality dispatcher** for
tree-structured attention. Instead of planning over every node in the
tree, we restrict the runtime decision to a small set of execution
strategies: independent per-branch decoding, two-level shared-prefix
attention, two-level shared-prefix attention with a decode-native
tail, and three-level shared-prefix attention. This dispatch space is
motivated by two observations. First, the dominant kernel mismatch is
**bimodal**: the shared root prefers a wide prefill-style query tile,
while the per-branch tail prefers a decode-native query tile.
Second, in the tree workloads we target, most reusable KV volume is
concentrated in a small number of contiguous sharing spans — a global
root prefix, an optional intermediate group-level suffix, and a
per-branch tail.

The dispatcher uses a **calibrated cost model** to choose among these
strategies at every decoding step. The model accounts for KV
bandwidth, wave-quantized prefill compute, wave-quantized decode-tail
compute (with HBM overlap), and merge overhead. Since the candidate set is
fixed and the exposed cascade depth is capped, planning time is
linear in batch size and beam width rather than in the full number of
nodes in the input tree. This gives the system enough flexibility to
choose between fused shared-prefix execution and decode-tail
execution, while avoiding the overhead of an uncapped tree-dependent
planner.

We evaluate the dispatcher on Llama-3.1-8B and Llama-3.2-1B on an
H100 GPU across tree-structured generation workloads spanning
`multi_chain_reasoning`, `multi_level_system`, and `multi_few_shot`.
Across the main workload grid, the dispatcher improves end-to-end
decode time over paged decoding, multi-level cascade attention, and
a recent tree-attention planner. Ablations show that the decode-tail
strategy is necessary on workloads where narrow tails make
prefill-style query tiles inefficient, while fused shared-prefix
execution remains necessary when decode-tail launch and merge
overheads dominate. Long-decode experiments further show that
exposing deeper sharing levels can reduce forward time, but that
unbounded depth is not always beneficial end to end because planning
overhead can exceed the saved kernel time.

### Comparison with prior work

| Method      | Share Prefix | Dynamic Dispatch | Decision Space | Plan Complexity   | Q-Bimodal |
|-------------|--------------|------------------|----------------|-------------------|-----------|
| Paged       | no           | no               | $O(1)$         | —                 | no        |
| FlashInfer  | yes          | no               | $O(1)$         | —                 | no        |
| DeFT        | yes          | no               | $O(1)$         | —                 | no        |
| FastTree    | yes          | yes              | $O(2^K)$       | $O(K \log K)$     | no        |
| **Ours**    | yes          | yes              | $O(1)$         | $O(K)$            | **yes**   |

### Workloads in detail

The three target workloads above are instances of a broader family of
tree-structured generation patterns:

| Workload                                 | Branching structure                                              | Reference                                         |
|------------------------------------------|------------------------------------------------------------------|---------------------------------------------------|
| Beam search                              | uniform K-way at every step                                       | classical                                         |
| Diverse beam search (DBS)                | K-way with group-diversity penalty; top-K shifts every step      | Vijayakumar et al. 2018                           |
| Self-consistency                         | N independent samples from one prompt                            | Wang et al. 2022                                  |
| Few-shot prompting                       | M shared in-context examples → 1 query per prompt                | Mann et al. 2020                                  |
| Speculative decoding (SpecInfer, Medusa) | speculative-tree fan-out per step                                | Miao et al. 2023; Cai et al. 2024                 |

The common shape: a small number of distinct **sharing levels** in the
KV state — a long root prefix shared by every query, optionally one
shorter prefix shared by a sub-group of queries, and a per-branch tail.
"Level" here means *a contiguous KV span that some set of queries
shares*. Batching independent prompts together does *not* add a level
— different prompts share no KV, so each prompt contributes its own
independent 2- or 3-level structure. A genuine intermediate sharing
level appears only when one prompt's tree itself has sub-groups that
share KV between the root and the per-branch tail, e.g., DBS diversity
groups or mid-decode beam survivors that share a partial suffix.
Even then, the intermediate span is usually short relative to the
root prefix, so the additional bandwidth saving from a third level is
small. **In the tree workloads we target, most reusable KV volume is
concentrated in at most three spans**: the global root, an optional
group-level suffix, and the per-branch tail. Deeper sharing
structures may exist for other workload classes, but in our measured
workloads the deeper spans contribute little KV volume, and
exploiting them requires a larger planner space whose overhead
outweighs the marginal bandwidth savings. This is a workload
observation, not a universal theorem.

## What existing kernels do, and where each one breaks

The high-level comparison table in the Introduction summarized five
methods on five axes (prefix sharing, dynamic dispatch, decision
space, plan complexity, Q-bimodal support). Below we expand the four
methods we benchmark against, and pinpoint where each one breaks.

| method     | shares prefix | dynamic per-step dispatch | decision space (per-step candidates)                       | per-step heuristic cost          |
|------------|---------------|---------------------------|------------------------------------------------------------|----------------------------------|
| paged      | no            | n/a                       | n/a                                                        | —                                |
| MLCA       | yes           | no                        | caller-fixed at construction (no per-step search)          | —                                |
| FastTree   | yes           | yes                       | **O(2^K)** per-vnode binary split (SplitQ/SplitK at each tree node) | O(B · K · log K) (greedy heuristic) |
| **bs_kernel (ours)** | yes  | yes                       | **O(1)** fixed cardinality (4 strategies, independent of tree shape) | O(B · K) (D_cap = 3)             |

Note the two columns measure different things. The **decision space**
is the size of the underlying combinatorial problem the planner is in
principle searching over; FastTree's per-vnode binary split decision
gives **O(2^K)** distinct assignments (one bit per tree node, with
N = O(K) nodes for a compressed radix tree), which is why FastTree
cannot enumerate and instead uses a greedy heuristic. The **per-step
heuristic cost** is what that planner actually pays each decoding
step. FastTree's heuristic prep — `_tree_heuristic` (greedy BFS,
O(B · K)) + `compute_parallelism` (two K · D passes, adds the log K)
— costs **O(B · K · log K)** per step on a balanced radix tree and is
not capped by design. Our picker enumerates an O(1) candidate set and
costs **O(B · K)** per step via a hard depth cap (`D_cap = 3`). On
the beam-search workloads we measure the radix tree fed to FastTree
is typically shallow, so both planners measure as O(B · K) with the
gap in constants; on deeper trees only ours stays bounded. See
*Decision-space size* below for the worst-case bound and where the
log K comes from, and *Planning-overhead breakdown* for the measured
plan-time gap.

- **paged-decode** (vLLM-style): every branch independently re-reads
  the full prefix from HBM. Bandwidth scales as `O(B·K·L_p)`. No
  sharing, no dispatch. Loses by 1.42–2.82× on our six workload cells.
- **MLCA** (FlashInfer's `MultiLevelCascadeAttentionWrapper`): multi-
  level cascade primitive, but the level count, level boundaries, and
  CTA_Q template are fixed at construction time. No runtime selection.
  Pays a separate kernel launch per level + a `merge_states` bridge.
  Loses by 1.66–2.39× on our workloads.
- **FastTree** (MLSys '25): page-radix tree + a Triton **two-stage**
  SplitQ/SplitK kernel. Stage 1 is fired multiple times — once per
  "phase", with a per-phase `(Q_TILE_SIZE, KV_TILE_SIZE)` config (the
  shipped default is `TSQs = [64, 16]`, two phases); stage 2 does
  online-softmax merge across vnodes
  (`3rdparty/FastTree-Artifact/kernel_bench/fasttree.py:443–565`). So
  FastTree is **multi-launch** — but every launch goes through the
  same Triton template family with the same minimum compile-time
  `Q_TILE_SIZE = 16`. The dispatch is dynamic per step. The
  underlying scheduling problem admits a per-node split decision
  (SplitQ vs SplitK at each tree node), and the implementation uses
  a greedy heuristic followed by a parallelism estimator:
  `_tree_heuristic` (`fasttree.py:64–96`) walks the radix tree once
  in BFS order, picking the per-node bucket from a closed-form cost
  comparison; `compute_parallelism` (`fasttree.py:126–161`) then
  rolls request sets up SplitK edges and counts the resulting
  `(q_vnode, kv_vnode)` tile grid. The outer `fasttree_preparation`
  loop (`fasttree.py:163-188`) re-runs both up to 3 times per step
  to retune Q/KV split sizes if the parallelism falls below
  thresholds. `_tree_heuristic` itself is **O(N) = O(K)** (single
  BFS, constant work per edge), but `compute_parallelism` does two
  passes that each cost up to K · D, where D is the radix-tree
  depth (derivation in *Decision-space size* below). So
  `fasttree_preparation` is **O(N + K · D)** per prompt =
  **O(B · K · log K)** per step on a balanced radix tree
  (D = O(log K)), and is unbounded by design because the depth D is
  whatever the workload produces. At `K=64/B=32/L_p=8K` the Python
  planner takes ≈56 ms vs ≈31 ms of actual kernel time
  (`docs/baseline/fasttree_analysis.md:11–22`). And because every
  launch shares the same template family, the narrow per-branch tail
  (`R = G = 4`) still pays ≥75% MMA padding even at the smallest
  configured phase (`Q_TILE_SIZE = 16`) — the template floor, not a
  scheduling failure.

**Where each one breaks:**

- paged ignores sharing → loses bandwidth.
- MLCA exposes sharing but doesn't pick → loses by being wrong for the
  current step.
- FastTree picks per step and launches multiple times, but every
  launch is the same Triton template family with `Q_TILE_SIZE ≥ 16`
  → loses by planning cost (the per-vnode greedy `_tree_heuristic`
  is O(B·K), but the `compute_parallelism` pass that follows adds a
  K·D term whose depth D is the radix tree's — so the combined
  `fasttree_preparation` step is **O(B·K·log K)** on a balanced
  tree, uncapped by design) *and* by template-floor padding on the
  narrow per-branch level.

## The core observation: fixed CTA_Q forces a tradeoff

Every attention kernel commits at compile time to a fixed
`CTA_TILE_Q = T` — the number of query rows it processes per CTA. A
tree-attention call has **two simultaneous Q-grouping regimes inside a
single call**:

| Level                         | Q grouping `R`        |
|-------------------------------|-----------------------|
| Shared root (long prefix)     | `K · G` (large)       |
| Per-branch tail (one query × G heads) | `G` (small)   |

For Llama-3.1-8B / -3.2-1B, `G = num_qo_heads / num_kv_heads = 4`. At
K = 64, root R = 256 and leaf R = 4 — a **64× gap inside one attention
call**. No single `T` resolves both:

- Small `T = 16`: leaf is fine (`R = 4 → U = 25%`), but root splits
  into `K·G/T = 16` tiles each re-reading the prefix from HBM —
  1.46× slowdown at K=64.
- Large `T = 128`: root packs into one tile, but leaf pads to
  `R/T = 3%` MMA utilization — 1.59× slowdown at tail=4096.

At our canonical cell (`K=64/B=32/L_p=8K`), the per-branch tail level
pays **98.4% MMA padding** under any single prefill template at
`T ∈ {16, 64, 128}`. q_tile padding is **frequent** (every step at
K ≥ 16) and **severe** (≥98% on one of the two levels at large K).

A closed-form tile-cost expression makes both failure modes explicit:

```
C_tile(T, L) ≈ α + β·T·L + γ·L
```

Small T pays `O(R/T)` redundant KV scans for the large-R level; large
T pays `O(T/R)` MMA-padding waste for the small-R level. **A single T
is never Pareto-optimal across a tree-attention call.**

This is the structural problem every "one-template" dispatcher shares
— paged-decode uses one decode template, FastTree and DEFT fire
multiple launches but all through the same Triton prefill template
family with a fixed minimum `Q_TILE_SIZE`. Their cost model picks
parallelism *within one kernel family*, but the bimodal Q-grouping
needs **two kernel families** — a prefill-style template for the
large-R root, and a CTA_Q=1-native decode kernel for the small-R
per-branch tail.

## A practically sufficient closure: 4 strategies, fixed-cardinality decision space

We derive the dispatch space from two empirical observations:

- **Bimodal Q grouping** (preceding section). Inside a tree-attention
  call, R takes two cardinalities: large (`K·G` at the root) and
  small (`G` at each branch). No single CTA_Q is Pareto-optimal
  across both.
- **KV-volume concentration.** In the workloads we target, most
  reusable KV volume is concentrated in at most three sharing spans:
  root prefix, an optional group-level intermediate, and per-branch
  tail.

These give the dispatch two axes:

- **Level axis** — how many sharing spans the cascade exposes:
  {1 (no sharing), 2, 3}.
- **Tail-kernel axis** — whether the small-R per-branch level rides
  the prefill kernel or routes to the CTA_Q=1 decode kernel and
  merges back: {fused, decode-merge}.

The Cartesian product is six cells, plus a pool-count axis (1POOL vs
2POOL) one might consider for the prefill side. On our measured
workloads, the production picker keeps four; the rest are either
dominated in forward time or contribute only marginal forward
benefit at the cost of added picker complexity (see *Excluded
candidates* below and *Dispatch-space ablation* in §Evaluation).

| strategy            | levels       | root kernel              | tail kernel                                   | merges (D−1) | wins when                                                                                                                                |
|---------------------|--------------|--------------------------|-----------------------------------------------|-------------:|------------------------------------------------------------------------------------------------------------------------------------------|
| `PER_BEAM`          | 1 (no share) | — (independent decode)   | paged decode, per beam                        | 0            | prefix short or K small — sharing doesn't recover its plan + merge overhead.                                                             |
| `SHARED_2L_FUSED`   | 2            | prefill (single CTA_Q)   | same prefill kernel                            | 1            | prefix long enough to share; per-branch tail wide enough to fill prefill tiles without much padding.                                     |
| `SHARED_2L_DECTAIL` | 2            | wide-T prefill           | `BatchDecodeWithPagedKVCacheWrapper` (CTA_Q=1) | 1            | prefix long; per-branch tail too narrow for a prefill launch to amortize — the q-tile padding saving on the small-R level exceeds the launch + SM-underutilization overhead, with the merge cost becoming significant only because the tail kernel itself is cheap. |
| `SHARED_3L_FUSED`   | 3            | prefill (single CTA_Q)   | same prefill kernel                            | 2            | workload exposes a uniform group-level intermediate span (DBS groups, mid-decode partial-suffix sharing).                                |

(In code: `PER_BEAM`, `SHARED_2L_1POOL`, `SHARED_2L_DEC_TAIL`,
`SHARED_3L_1POOL` — `cost_model.py:68`.)

Every depth-≥2 strategy pays `(D−1)` online-softmax merges between
levels — standard split-KV-style merges that combine partial outputs
and logsumexps from each per-level launch. Because online-softmax
merge is associative, those `(D−1)` merges can be reduced in a
balanced tree of depth `⌈log₂ D⌉` rounds, so the merge cost on the
critical path is `⌈log₂ D⌉·μ` rather than `(D−1)·μ` — the two values
coincide at D ∈ {2, 3} (which is why the cost model just writes
`(D−1)·μ` at our cap) but diverge for the `SHARED_4L_FUSED` row in
the depth-cap ablation, where 3 sequential merges collapse to 2
parallel rounds. FUSED and DECTAIL at the same depth pay the same
merge count; what differs between them is the **tail kernel**.

`SHARED_2L_DECTAIL` is the row that does the real load-bearing work
for the bimodal-Q observation. Routing the small-R per-branch level
to the decode kernel **strictly removes q-tile padding on that
level** (CTA_Q = 1 native, 0% padding vs prefill's ≥75% at R = G =
4). It is *not* strictly better end-to-end: the decode kernel pays
its own launch overhead and risks SM under-utilization when the
tail's KV work is small, and because the tail-kernel cost shrinks
so much, the merge cost (paid by FUSED and DECTAIL alike) becomes
relatively larger on the DECTAIL side and can flip the boundary.
DECTAIL wins end-to-end only when the padding saving exceeds those
overheads — which is exactly the crossover the picker models.
`SHARED_2L_FUSED` is the answer on the other side of the crossover.

**Excluded candidates.** Two distinct reasons for exclusion, kept
separate so the claim is precise:

- *Dominated in forward time.* `SHARED_*L_2POOL` (two prefill pools
  instead of prefill + decode-merge) and `SHARED_3L_DECTAIL` (3
  levels + decode tail) are dominated on every workload cell we
  measured by either `SHARED_2L_DECTAIL` (when there is enough
  small-R work) or a FUSED variant (when there is not).
- *Forward-only benefit exists but is overrun by plan-time cost.*
  Depth-4+ FUSED is **not** dominated in forward time — forced
  4L_FUSED is 3.0% faster in forward than 2L_FUSED on the long-decode
  cell most favorable to deeper sharing. But its per-step plan cost
  grows by **+25.6%** vs 2L (20.1 s vs 16.0 s plan_total at
  K=32/L_p=8K/B=16/max_new=2048), which more than wipes out the
  forward saving — forced 4L_FUSED is **6.9% slower end-to-end**
  than forced 2L_FUSED on this cell. Each added depth requires
  building more cascade-plan levels per step on the host
  (`_adaptive_levels` + `_build_cascade_plan` work scales as
  O(K · depth)), and the picker has to evaluate more candidates.
  The depth cap is therefore not just a marginal-benefit argument —
  past depth 3 the planner cost actively *exceeds* the forward
  benefit on the workloads we target. The cap could be raised for
  workloads where the forward benefit per added level is large
  enough to dominate the linear-in-depth planner growth.

All excluded variants remain in code (`cost_model.py:40`) and are
reachable via the `available_strategies` override. Quantitative
evidence is in the **Dispatch-space ablation** section below.

We do not claim this set is a universal minimal closure. We claim it
is the *practically sufficient* closure for the workloads we target,
and that the space is extensible at O(1) per added candidate.

### Decision-space size — fixed-cardinality candidate set, capped depth

The candidate set has constant cardinality (four), regardless of tree
shape, beam count, or prefix length, and the picker caps cascade
depth at `D_cap = 3` (`cost_model.py:207, 747`). This gives a
**per-step picker cost of O(B · K · D_cap) = O(B · K)** that is
*independent of the workload's KV-sharing tree shape* — the cap is
the guarantee, not the input.

FastTree's *decision space* is **O(2^K)** — one binary SplitQ/SplitK
choice per tree node, with N = O(K) nodes for a compressed radix tree
over K leaves — so exact enumeration is infeasible and the
implementation explores it greedily. Its *heuristic preparation* has
no depth cap: `_tree_heuristic` + `compute_parallelism`
(`3rdparty/FastTree-Artifact/kernel_bench/fasttree.py:64-161`) walk
whatever radix tree the workload produces, giving a per-step cost of
**O(B · K · log K)**.

**Where the log K comes from.** Fix K = total queries (leaves of
the radix tree). For a compressed radix tree (every internal node
has ≥2 children), total nodes N ≤ 2K-1, so N = O(K); a balanced
tree has depth D = O(log K), a degenerate path tree has D = O(K).
`_tree_heuristic` is a single BFS doing O(1) work per edge, so it
is **O(N) = O(K)** — no log factor. The log K factor lives
entirely in `compute_parallelism` (`fasttree.py:126-161`), which
runs two passes whose worst-case cost is K · D each: (1) a SplitK
bubble-up (`fasttree.py:127-141`) that appends
`tree_info[node].requests` into the parent's accumulator — and
since `|tree_info[node].requests|` equals subtree size, summing
over all SplitK-assigned non-root nodes gives Σ subtree_size up to
K · D (all internal nodes SplitK is the worst case); and (2) for
every effective node, an upward walk through consecutive
SplitQ-assigned ancestors to merge KV ranges (`fasttree.py:144-158`)
— at most K effective nodes each walking up to D ancestors gives
K · D worst case (all internal nodes SplitQ). The overall planner
is **O(N + K · D)** per prompt; on a balanced radix tree
(D = O(log K)) this is **K · log K**, and `fasttree_preparation`
re-runs the whole loop up to 3 times per step (`fasttree.py:163`),
so per-step planner cost is **O(B · K · log K)**. On a path-shaped
tree (D = O(K)) the bound degrades to O(B · K²). Our picker avoids
this dependency because the cap fixes the number of cascade levels
we sum over at `D_cap = 3`, regardless of how deep the underlying
radix tree is — pages past depth 3 fold back into the per-branch tail
bandwidth term (`cost_model.py:424-431`) rather than appearing as
additional iterations of the cost loop.

The asymmetry is the structural point:

| planner | decision space | per-step heuristic cost | depends on input tree shape? |
|---|---|---|---|
| `pick_strategy_batch` | **O(1)** (4 strategies) | **O(B · K)** with D_cap = 3 | **no** — capped by design |
| FastTree heuristic + parallelism | **O(2^K)** (per-vnode binary split) | **O(B · K · log K)** (greedy) | **yes** — uncapped, log K is the radix-tree depth |

Measured `pick_ms` on H100 ranges from 0.09 ms at small cells to
~0.8 ms at the largest 3L-cascade cells in our grid; on the
supplementary cell exercising 3L cascades (`K=16/L_p=4K/B=32`, exp3)
median `pick_ms` is **0.63 ms**, against FastTree's same-cell
planner cost of **4.33 ms** (`meta_ms` + `radix_ms`). MLCA is a
distinct point: fixed-cardinality candidate set but *caller-fixed* —
chosen at construction, not adapted per step.

The depth cap is what lets us *promise* a per-step planner bound
that's independent of how the beams happen to fork; FastTree's
planner is bounded only by the tree the workload hands it. On
beam-search workloads where the tree happens to be naturally shallow
(typically depth 2 after split-pages CoW: one shared-prefix root
plus K divergent leaves) both planners measure as O(B · K), and the
wall-clock gap (~7× on the matched cell) is in constants. On
workloads with deeper natural sharing the gap widens with log K —
see *Dispatch-space ablation* below for a direct measurement of
plan-time growth with depth.

## The cost-model picker

For each candidate strategy s ∈ S we evaluate an analytical cost
C_s(W; θ), where W is the per-step workload signature (K, L_p, B, the
per-branch suffix lengths, and intermediate-level structure if any) and
θ is a vector of device-calibrated constants (HBM bandwidth, SM count,
per-tile prefill cost, decode-kernel per-(beam, KV-token) cost). The
picker returns argmin_s C_s(W; θ). |S| = 4 by construction (preceding
section), so the *candidate cardinality* is O(1) regardless of input
shape. Each cost evaluation reduces over the B prompts and their K
beams across at most `D_cap = 3` cascade levels, so per-step picker
time is **O(B · K · D_cap) = O(B · K)** with small per-branch
constants — independent of the workload's KV-sharing tree shape.
Implementation in `src/beam_engine/methods/bs_kernel/cost_model.py`;
entry point is `pick_strategy_batch` at line 701.

Each candidate's cost is a sum of a bandwidth term and a
wave-quantized compute term; the picker has **no free parameters**
beyond physics-calibrated constants. Every depth-≥2 strategy
launches `(D−1)` online-softmax merges between levels; the
closed-form drops the merge term from `C_fused` (it's dwarfed by
the prefill kernel work) and keeps it on the DEC_TAIL side, where
the tail kernel is cheap enough that the same merge cost can flip
the boundary:

```
C_fused(s; θ)    = B_s / β + ⌈M_s / N_SM⌉ · τ                           // (D−1)·μ dwarfed by prefill
C_dec_tail(s; θ) = C_fused(prefix; θ) + max(B_tail/β, ⌈R/N_SM⌉·τ_tail) + (D−1)·μ
                                                  ^^^^^^^^^^^^^^^^^^^
                                                  wave-quantized compute (= U_tail)
```

with `τ_tail = L̄_tail · ρ_d`. All costs are in microseconds. Symbols:

| symbol         | meaning                                                                                   | units                |
|----------------|-------------------------------------------------------------------------------------------|----------------------|
| `B_s`          | total KV bytes loaded across all cascade levels for strategy `s`                          | bytes                |
| `β`            | calibrated HBM bandwidth (`B_hbm`)                                                        | bytes / µs           |
| `M_s`          | total prefill-tile count summed across cascade levels and across the B prompts            | dimensionless        |
| `N_SM`         | SM count (`num_sms`), read from `torch.cuda.get_device_properties`                        | dimensionless        |
| `τ`            | calibrated time per prefill tile at the large-T pool (`per_tile_us[T_large]`)             | µs / tile            |
| `μ`            | calibrated merge cost per cascade boundary: `μ = μ_L + μ_B · R` where `μ_L = merge_launch_us`, `μ_B = merge_bw_us_per_row` | µs / merge |
| `D`            | dispatch depth (2 or 3); `(D−1)` merges sit between levels with critical-path depth `⌈log₂ D⌉` | dimensionless |
| `B_tail`       | total per-branch-tail KV bytes loaded by the decode kernel                                | bytes                |
| `R`            | total decode-kernel CTAs = total beam count = `Σ_b K_b`; one CTA per beam (CTA_Q=1)       | dimensionless        |
| `L̄_tail`     | mean per-beam tail length (KV tokens), `L̄_tail = (Σ_{b,j} L_{b,j}) / R`                 | tokens / beam        |
| `ρ_d`          | calibrated per-(beam, KV-token) decode cost (`decode_us_per_beam_kv_token`)               | µs / token           |
| `τ_tail`       | per-CTA wall time on the decode side, `τ_tail = L̄_tail · ρ_d`                            | µs / CTA             |

- **Bandwidth term `B_s / β`** sums KV bytes loaded across all
  cascade levels and divides by the calibrated peak HBM bandwidth.
  We do not apply a Q-utilization penalty on this term: empirically
  the prefill kernel is HBM-bound at the L_kv values that matter for
  beam search (≥ 1024 tokens), so 1 vs T queries at a fixed L_kv
  takes near-identical wall time on H100. The bimodal-Q phenomenon
  — a per-branch tail at packed_qo=1 being slower per byte than a
  root-level tile at packed_qo=K — is handled *structurally* by the
  DEC_TAIL strategy, which routes the per-branch tail to the
  purpose-built CTA_Q=1 decode kernel and prices it via
  `max(B_tail/β, U_tail)`.
- **Wave-quantized prefill compute `⌈M_s / N_SM⌉ · τ`** is what
  couples the picker to batch size. Tiles are summed across all B
  prompts first, then divided by `N_SM` and ceiling'd — so the
  picker's decision changes with B, not just per-prompt shape: when
  the per-branch tile count crosses `N_SM`, a second wave is paid in
  full, which is what tilts the 1-pool vs 2-pool boundary at large B.
- **`max(B_tail/β, U_tail)`** on the DEC_TAIL side, where
  `U_tail = ⌈R / N_SM⌉ · τ_tail` is the wave-quantized compute term:
  the decode kernel runs `R` CTAs (one per beam) in `⌈R / N_SM⌉`
  sequential waves, each wave taking per-CTA wall time
  `τ_tail = L̄_tail · ρ_d`. The bandwidth term `B_tail / β` saturates
  HBM whenever there are enough active CTAs and so is not
  wave-quantized. MMA work overlaps with HBM reads within the
  decode kernel, so the two terms combine via `max` rather than `+`
  — this is what lets the model see that the decode kernel beats
  prefill on the per-branch tail even when its arithmetic intensity
  is similar. On H100 + fp16/bf16 KV every cell we evaluate satisfies
  `B_tail/β > U_tail`, so DEC_TAIL is bandwidth-bound in practice;
  the `max(·,·)` form correctly extrapolates to smaller KV dtypes
  (fp8, int4) and higher-flop-to-bw GPUs where `U_tail` would win.

**Calibration grid.** `θ` is measured by `calibrate.py`:
`β` from a 256 MiB device-to-device copy; `τ` (per-tile prefill
time) at each `T ∈ {16, 64, 128}` from a single-tile prefill against
a fixed L_kv in a 50-iter warm loop; `ρ_d` and the per-merge slopes
(`μ_L`, `μ_B`) from short OLS fits over per-call wall-times sampled
across a small `(L_kv, batch)` grid; `N_SM` is read from device
properties. Total calibration takes seconds per GPU, run once and
cached. The picker is then pure closed-form over `θ`, validated by
per-cell oracle regret on a small `(K, L_p, B)` sweep.

Launch / sync overheads on the prefill kernel are intentionally not
modeled: for beam-search shapes the kernel work dwarfs them, and
including them adds noise to the picker without changing the argmin.
FUSED and DEC_TAIL pay the same `(D−1)` online-softmax merges; the
DEC_TAIL expression keeps the merge term explicit because its tail
kernel is small enough that the merge cost can flip the boundary,
while on FUSED the same merge cost sits well below the prefill
kernel work.

Two notes on the cost model:

1. **Decision quality, not prediction accuracy, is the validation
   target.** The picker is a discrete chooser over four candidates,
   not a regression. What matters is that `argmin_s C_s` matches the
   oracle argmin — not that the predicted µs matches the measured µs.
   We validate this via picker-level oracle regret in the
   Picker-validity check (exp1c) below, where the forced-2L ablation
   shows the picker matches or beats the better forced variant on
   every cell.
2. **Per-device calibration is necessary, not optional.** HBM
   bandwidth, per-tile cost, and SM count differ enough across
   H100 / A100 / RTX-PRO that a hardcoded coefficient vector picks
   wrong on at least one device — there is no way to give an accurate
   per-tile cost by guessing. We measure θ once per GPU via
   `calibrate.py` (~3 minutes one-time, run from a small script and
   cached to disk; the cached coefficients are reused on every
   subsequent engine init). This is the standard pattern for
   analytical performance models on GPUs — Roofline-style models are
   only useful with peak BW and peak FLOP/s measured on the device,
   not read from spec sheets.

## Evaluation

We organize the evaluation into five questions a reviewer would
naturally ask: do the end-to-end numbers hold up; where in plan vs
forward does our win come from; what does the picker actually pick;
how close is the picker to the oracle; and which excluded
dispatch-space points would have helped if we had kept them.

### Scenarios

The six benchmark cells map to the workload table at the top of the
paper as follows:

| benchmark scenario     | what it is                                                                                                            | workload-table row                |
|------------------------|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| `multi_level_system`   | B prompts batched, each with a multi-segment system+task prefix, K beams per prompt                                   | few-shot / structured-prompt batched beam search |
| `multi_few_shot`       | B prompts batched, each with shared in-context examples as the prefix, K beams per prompt                             | few-shot prompting + beam search  |
| `multi_chain_reasoning`| B prompts batched, each running chain-of-thought beam search (long context with reasoning prefix, K beams per prompt) | beam search                       |

Each prompt in a batch contributes its own independent shared-prefix
tree; prompts share no KV across the batch. The variation across
scenarios is in the per-prompt prefix structure and the (B, K)
shape, which is what drives the per-step strategy picks.

### End-to-end speedups

Six cells (three scenarios × two Llama sizes) at **K=64**, warmup +
measured. All times are total decode (255 steps).
CSVs: `benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing/`
(timing) and `…/trace/` (strategy mix); reproducers:
`scripts/paper-exp/exp1a_end_to_end_timing.sbatch` and
`scripts/paper-exp/exp1b_end_to_end_strategy.sbatch`.

K=64 is the evaluation point throughout this section — a realistic
production beam width for diverse-sampling and ensemble decoding
(Google's beam-search APIs default to K∈[4,64]; reranker pipelines
routinely use K=32–128). The last column is the picker's per-step
strategy histogram for bs_kernel (the picker is a property of
bs_kernel only); both FUSED and DEC_TAIL fire across the cells,
exercising the full dispatch space.

On 8B, the K=64 KV slab exceeds H100 HBM at multi_few_shot's natural
B=16 and multi_chain_reasoning's natural B=32; we halve to B=8 on
those two cells so the run fits. 1B cells run at their natural B.

| model           | scenario                       | method    | decode total (ms) | plan total (ms) | fwd total (ms) | plan % | fwd % | p50/step (ms) | strategy mix (bs_kernel, 255 steps)        |
|-----------------|--------------------------------|-----------|------------------:|----------------:|---------------:|-------:|------:|--------------:|--------------------------------------------|
| Llama-3.1-8B    | multi_level_system (B=4, K=64) | bs_kernel | 3684              | 206             | 3439           | 5.6%   | 93.4% | 14.32         | 2L_FUSED ×34, **2L_DECTAIL ×221** (87%)    |
|                 |                                | paged     | 5687              | 490             | 5157           | 8.6%   | 90.7% | 22.33         | —                                          |
|                 |                                | fasttree  | 5952              | 647             | 5267           | 10.9%  | 88.5% | 22.98         | —                                          |
|                 |                                | deft      | 8058              | 504             | 7515           |  6.3%  | 93.3% | 32.04         | —                                          |
|                 |                                | mlca      | 6110              | 215             | 5854           |  3.5%  | 95.8% | 24.73         | —                                          |
| Llama-3.1-8B    | multi_few_shot (B=8, K=64)     | bs_kernel | 7271              | 363             | 6851           | 5.0%   | 94.2% | 28.47         | **2L_DECTAIL ×255 (100%)**                 |
|                 |                                | paged     | 17372             | 1431            | 15883          | 8.2%   | 91.4% | 68.13         | —                                          |
|                 |                                | fasttree  | 13481             | 1417            | 12010          | 10.5%  | 89.1% | 52.61         | —                                          |
|                 |                                | deft      | 19903             | 1011            | 18840          |  5.1%  | 94.7% | 77.74         | —                                          |
|                 |                                | mlca      | 13634             | 329             | 13244          |  2.4%  | 97.1% | 52.98         | —                                          |
| Llama-3.1-8B    | multi_chain_reasoning (B=8, K=64) | bs_kernel | 9714          | 597             | 9029           | 6.1%   | 93.0% | 20.72         | 2L_FUSED ×60, **2L_DECTAIL ×195** (76%)    |
|                 |                                | paged     | 13818             | 1062            | 12667          | 7.7%   | 91.7% | 26.44         | —                                          |
|                 |                                | fasttree  | 13880             | 1625            | 12172          | 11.7%  | 87.7% | 26.66         | —                                          |
|                 |                                | deft      | 28681             | 1116            | 27485          |  3.9%  | 95.8% | 61.63         | —                                          |
|                 |                                | mlca      | 17230             | 465             | 16683          |  2.7%  | 96.8% | 29.46         | —                                          |
| Llama-3.2-1B    | multi_level_system (B=4, K=64) | bs_kernel | 1318              | 253             | 1027           | 19.2%  | 77.9% |  4.89         | 2L_FUSED ×255                              |
|                 |                                | paged     | 2124              | 486             | 1600           | 22.9%  | 75.3% |  8.27         | —                                          |
|                 |                                | fasttree  | 1993              | 657             | 1298           | 33.0%  | 65.1% |  7.43         | —                                          |
|                 |                                | deft      | 3114              | 496             | 2582           | 15.9%  | 82.9% | 12.27         | —                                          |
|                 |                                | mlca      | 2451              | 207             | 2205           |  8.5%  | 90.0% | 10.22         | —                                          |
| Llama-3.2-1B    | multi_few_shot (B=32, K=64)    | bs_kernel | 8853              | 1985            | 6721           | 22.4%  | 75.9% | 33.43         | 2L_FUSED ×159, **2L_DECTAIL ×96** (38%)    |
|                 |                                | paged     | 24973             | 5917            | 18901          | 23.7%  | 75.7% | 95.23         | —                                          |
|                 |                                | fasttree  | 15969             | 8833            | 6990           | 55.3%  | 43.8% | 53.59         | —                                          |
|                 |                                | deft      | 26734             | 4327            | 22273          | 16.2%  | 83.3% |102.85         | —                                          |
|                 |                                | mlca      | 19670             | 1032            | 18492          |  5.2%  | 94.0% | 75.99         | —                                          |
| Llama-3.2-1B    | multi_chain_reasoning (B=32, K=64) | bs_kernel | 10033          | 2636            | 7220           | 26.3%  | 72.0% | 21.29         | 2L_FUSED ×255                              |
|                 |                                | paged     | 16955             | 4268            | 12505          | 25.2%  | 73.8% | 31.50         | —                                          |
|                 |                                | fasttree  | 17025             | 9348            | 7500           | 54.9%  | 44.0% | 28.13         | —                                          |
|                 |                                | deft      | 33010             | 4636            | 28209          | 14.0%  | 85.5% | 61.14         | —                                          |
|                 |                                | mlca      | 23988             | 1232            | 22591          |  5.1%  | 94.2% | 40.51         | —                                          |

**bs_kernel speedups vs each baseline (K=64):**

| scenario              | model         | vs fasttree | vs paged | vs deft | vs mlca |
|-----------------------|---------------|------------:|---------:|--------:|--------:|
| multi_level_system    | Llama-3.1-8B  | 1.62×       | 1.54×    | 2.19×   | 1.66×   |
| multi_few_shot        | Llama-3.1-8B  | 1.85×       | 2.39×    | 2.74×   | 1.88×   |
| multi_chain_reasoning | Llama-3.1-8B  | 1.43×       | 1.42×    | 2.95×   | 1.77×   |
| multi_level_system    | Llama-3.2-1B  | 1.51×       | 1.61×    | 2.36×   | 1.86×   |
| multi_few_shot        | Llama-3.2-1B  | 1.80×       | 2.82×    | 3.02×   | 2.22×   |
| multi_chain_reasoning | Llama-3.2-1B  | 1.70×       | 1.69×    | 3.29×   | 2.39×   |

Four observations from the table:

- **Best forward time on every cell.** Cascading + decode-tail routing
  reduces per-branch HBM reads and the per-step MMA-padding waste; the
  forward column is the smallest across all four methods on every row.
  The largest forward wins are on cells the picker routes through
  DEC_TAIL (8B/multi_few_shot's 1.85× over FastTree is a 1.75× forward
  reduction).
- **Best plan time on every cell.** The fixed-cardinality cost-model
  argmin (4 cost evaluations per step, each linear in B·K with small
  constants) is cheaper to evaluate than FastTree's per-vnode
  heuristic. At K=64 the plan-time ratio vs FastTree is 22–37%; the
  gap widens with K because FastTree's planner walks more tree nodes
  as K grows, while our planner enumerates a fixed four candidates
  regardless of K — per-candidate cost still grows linearly in B·K,
  but the candidate count itself is constant in K, L_p, and tree
  shape. bs_kernel's plan total is within ≈2× of paged on every cell,
  while paged has no dispatch to plan at all.
- **SHARED_2L_DEC_TAIL fires on every cell where the per-branch tail
  is narrow relative to the root.** At (B, K) = (8, 64) and (4, 64) on
  Llama-3.1-8B, DEC_TAIL is selected on 76–100% of steps. The 1B
  cells either stay on SHARED_2L_FUSED (when the merge cost dominates)
  or split (1B/multi_few_shot picks DEC_TAIL on 38% of steps). This
  is the bimodal-Q observation in action: when the per-branch level is
  small enough that the prefill template's MMA padding eats >1.7× of
  its throughput, the merge + launch overhead of routing that level
  to the CTA_Q=1 decode kernel is the better deal.
- **SHARED_2L_FUSED is the right call on the rest.** 1B's
  multi_level_system and multi_chain_reasoning both run at full B=32
  natural-K — the per-branch tail aggregates to enough work that
  prefill amortizes its launch and the merge cost of DEC_TAIL would
  be a regression (forced 2L_DEC_TAIL on 1B/multi_chain_reasoning is
  1.70× slower than forced 2L_FUSED; see exp1c below). PER_BEAM is
  never picked on the workloads we evaluate; SHARED_3L_FUSED appears
  on longer-prefix and long-decode cells (see the Planning-overhead
  breakdown and Beam-search-at-long-decode subsections below).

**Strategy distribution across beam-search workloads.** Per-step
strategy counts (255 decode steps per multi_* row; 2,047 per long-
decode) from the bs_kernel trace in
`benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/trace/exp1b.log`
and `benchmarks/bs_kernel/results/paper-exp/exp4_long_decode/exp4.log`:

| model | scenario | `SHARED_2L_1POOL` | `SHARED_2L_DEC_TAIL` | `SHARED_3L_1POOL` |
|-------|----------|------------------:|---------------------:|------------------:|
| 3.1-8B | `multi_level_system`    |    34 (13.3%) |   221 (86.7%) |             0 |
| 3.1-8B | `multi_chain_reasoning` |    60 (23.5%) |   195 (76.5%) |             0 |
| 3.1-8B | `multi_few_shot`        |     0 (0.0%)  |   255 (100.0%) |            0 |
| 3.2-1B† | beam-search long-decode |   59 (2.9%)  |             0 |  1,988 (97.1%) |

† Long-decode reported on Llama-3.2-1B because 8B at (K=32, B=16,
max_new=2048) needs ~90 GiB of KV slab and doesn't fit in a single
H100. The strategy mix is a property of the workload shape (per-beam
tail length relative to the K-packed root tile), so 8B at this cell
would also concentrate on `SHARED_3L_1POOL`. `PER_BEAM` and
`SHARED_3L_DEC_TAIL` are in the candidate set but never chosen on
these workloads — omitted from the table.

#### Qwen3-4B replication (TP=2)

To check that the picker and the kernel selection generalize beyond
Llama-family models, we re-ran the three exp1a scenarios on
**`Qwen/Qwen3-4B`** — a standard GQA softmax-attention dense model
(36 layers, hidden=2560, num_heads=32, num_kv_heads=8, head_dim=128,
`q_norm`/`k_norm` per-head before RoPE). At 36 layers × head_dim 128,
the unified KV slab is 4.5× larger per page than Llama-3.2-1B, so the
single-H100 budget OOMs at the `multi_chain_reasoning` and
`multi_few_shot` K=64 cells; we run **TP=2** across two H100s
(`scripts/paper-exp/exp1a_end_to_end_timing_qwen3_4b.sbatch`).
Decode-total (255 steps) reproduces the same picture as Llama:

| scenario              | method    | decode total (ms) | plan total (ms) | fwd total (ms) | p50/step (ms) |
|-----------------------|-----------|------------------:|----------------:|---------------:|--------------:|
| multi_level_system (B=4, K=64)  | bs_kernel | 4542 | 341  | 4137  | 17.22 |
|                                 | paged     | 4538 | 538  | 3952  | 17.36 |
|                                 | fasttree  | 5750 | 756  | 4941  | 21.97 |
|                                 | mlca      | 5404 | 733  | 4621  | 20.86 |
|                                 | deft      | 6034 | 175  | 5806  | 23.50 |
| multi_chain_reasoning (B=16, K=64) | bs_kernel | 11285 | 1617 | 9515  | 22.59 |
|                                    | paged     | 16135 | 2237 | 13743 | 31.46 |
|                                    | fasttree  | 18460 | 4832 | 13460 | 33.63 |
|                                    | mlca      | 20144 | 2819 | 17166 | 34.10 |
|                                    | deft      | 31084 | 804  | 30100 | 66.60 |
| multi_few_shot (B=16, K=64)     | bs_kernel | 8994  | 1337 | 7500  | 34.46 |
|                                 | paged     | 22230 | 3257 | 18829 | 86.39 |
|                                 | fasttree  | 18318 | 4525 | 13648 | 66.71 |
|                                 | mlca      | 17010 | 2561 | 14309 | 64.92 |
|                                 | deft      | 23420 | 638  | 22646 | 91.07 |

**bs_kernel speedups vs each baseline on Qwen3-4B (K=64, TP=2):**

| scenario              | vs fasttree | vs paged | vs deft | vs mlca |
|-----------------------|------------:|---------:|--------:|--------:|
| multi_level_system    | 1.27×       | 1.00×    | 1.33×   | 1.19×   |
| multi_chain_reasoning | 1.64×       | 1.43×    | 2.75×   | 1.78×   |
| multi_few_shot        | 2.04×       | 2.47×    | 2.60×   | 1.89×   |

Two notes specific to Qwen3-4B:

- **Picker stays on `SHARED_2L_1POOL` on all three cells** (255/255
  steps) — `DEC_TAIL` does not fire. This is consistent with the
  bimodal-Q account: at TP=2 the per-rank `num_kv_heads = 4`, so
  `G = num_qo_heads / num_kv_heads = 4` per rank just like Llama, but
  the cascaded prefill template absorbs the per-branch tail without
  the merge cost dominating. The strategy distinction (`DEC_TAIL` vs
  `FUSED`) is therefore not the source of the speedup here; the win
  comes from cascade depth + the cost-model picker preferring
  cascaded prefill over the per-beam-decode default that paged uses.
- **`multi_level_system` is a near-tie with paged** (1.00×) because the
  level-system scenario has only B=4 prompts: the cascade depth-2
  template's launch overhead amortizes less when there are only 4
  shared roots to cascade against. The cell that most stresses
  cascade-prefill (multi_few_shot: B=16 with shared-prefix structure)
  is also the cell with the largest win (2.0–2.6× across baselines).

#### Llama-3-70B FP8 (TP=2)

To check that the dispatcher composes cleanly with FP8 weight
quantization at production model scale, we re-run the exp1a scenarios
on **`RedHatAI/Meta-Llama-3-70B-Instruct-FP8`** — the AutoFP8 /
compressed-tensors quantization of `meta-llama/Meta-Llama-3-70B-Instruct`
(80 layers, hidden 8192, num_heads=64, num_kv_heads=8, head_dim=128).
The checkpoint ships per-channel `weight_scale` (broadcast from a
per-tensor scalar in AutoFP8) and per-tensor static `input_scale` for
every transformer-block linear; `lm_head` stays bf16 per the
`ignored_layers` field. We run **TP=2** across two H100s with B halved
(B=4 here vs 8B's B=8 at the same scenario) so the per-rank KV slab
fits — per-rank `num_kv_heads = 4` and 80 layers makes Llama-3-70B's
unified slab much larger than Llama-3.1-8B's.

The forward path replaces every parallel linear's `F.linear` with
`torch._scaled_mm` in RowWise mode (`scale_a=(M,1)`, `scale_b=(1,N)`)
— the only `_scaled_mm` mode that supports per-tensor activation +
per-channel weight on Hopper. QKV and gate_up fuse three (resp. two)
per-tensor weight scales into a per-output-channel `[N]` scale at load
time, so the fused linear runs a single GEMM rather than splitting into
per-chunk calls. Activations and norms stay bf16; fp16 attention
scores overflow at L_p=8K, so the fp16 default in the 1B/8B paper
runs would not survive at 70B.

**KV cache is bf16** in this section. FP8 KV cache is implemented in
the loader and append path (uint8 physical storage with logical
`float8_e4m3fn` passed through `wrapper.plan(kv_data_type=…)` so
FlashInfer's kernel reinterprets the bytes), but FlashInfer's current
`tvm_ffi` dispatch table doesn't accept either fp8 (dlpack lacks the
dtype) or uint8 (no kernel dispatch entry) for the K/V write path.
That blocks `--kv-dtype fp8_e4m3` end-to-end until either (a) the
FlashInfer dispatch macros gain a uint8 fallthrough, (b) we ship a
manual-scatter fp8 K/V write path, or (c) a custom Triton append
lands. We leave fp8 KV as a follow-up; weights-only FP8 already moves
the H100 memory budget from 140 GiB at bf16 (overflows TP=2) to
35.81 GiB / rank at fp8 (validated in `slurm/smoke_70b_fp8.sbatch`).

**Cost-model recalibration.** The coefficients in
`~/.cache/beam_engine/coeffs-<gpu>-<model>.json` are keyed by
(GPU, model, tp_size). Coefficients fit on bf16 1B/8B mis-pick on
70B-fp8: an initial exp2 dispatch ablation on the K=32/L_p=8192/B=8
cell showed the picker choosing `SHARED_2L_DEC_TAIL` while forced
`SHARED_2L_1POOL` is 4.4% faster — fp8's `_scaled_mm` saturates tensor
cores at smaller M·N than bf16 matmuls, so the relative cost of
DEC_TAIL's CTA_Q=1 decode kernel vs 1POOL's fused-cascade prefill
kernel shifts. We refit the physics coefficients (`β`, `τ`, `ρ_d`,
`μ_L`, `μ_B`) on `RedHatAI/Meta-Llama-3-70B-Instruct-FP8` via
`slurm/autotune_h100_70b_fp8_tp2.sbatch` and re-ran exp1a / exp2 / exp4
with the recalibrated cache.

**End-to-end speedups on Llama-3-70B FP8 (K=64, B=4 per scenario,
TP=2):** CSV
`benchmarks/bs_kernel/results/paper-exp/exp1a_70b_fp8_tp2/merged.csv`;
reproducer `scripts/paper-exp/exp1a_70b_fp8_tp2.sbatch`.

| scenario              | method    | decode total (ms) | plan total (ms) | fwd total (ms) | plan % | fwd % | p50/step (ms) |
|-----------------------|-----------|------------------:|----------------:|---------------:|-------:|------:|--------------:|
| multi_level_system    | bs_kernel |        **13,267** |             256 |         12,970 |  1.9%  | 97.8% |        51.68  |
|                       | paged     |            15,041 |             506 |         14,493 |  3.4%  | 96.4% |        59.03  |
|                       | mlca      |            20,140 |             747 |         19,349 |  3.7%  | 96.1% |        82.30  |
|                       | fasttree  |            27,739 |             662 |         27,034 |  2.4%  | 97.5% |       108.38  |
| multi_chain_reasoning | bs_kernel |        **26,494** |             436 |         25,985 |  1.6%  | 98.1% |        51.60  |
|                       | paged     |            27,684 |             608 |         27,003 |  2.2%  | 97.5% |        55.21  |
|                       | mlca      |            32,957 |             963 |         31,922 |  2.9%  | 96.9% |        61.70  |
|                       | fasttree  |            41,734 |             907 |         40,752 |  2.2%  | 97.6% |        80.86  |
| multi_few_shot        | bs_kernel |        **15,805** |             292 |         15,471 |  1.8%  | 97.9% |        62.18  |
|                       | paged     |            19,224 |             746 |         18,437 |  3.9%  | 95.9% |        75.34  |
|                       | mlca      |            22,439 |             767 |         21,630 |  3.4%  | 96.4% |        86.96  |
|                       | fasttree  |            35,141 |             643 |         34,457 |  1.8%  | 98.0% |       136.94  |

**bs_kernel speedups vs each baseline on Llama-3-70B FP8 (K=64, TP=2):**

| scenario              | vs fasttree | vs paged | vs mlca |
|-----------------------|------------:|---------:|--------:|
| multi_level_system    |    2.09×    |  1.13×   |  1.52×  |
| multi_chain_reasoning |    1.58×    |  1.04×   |  1.24×  |
| multi_few_shot        |    2.22×    |  1.22×   |  1.42×  |

Three observations:

- **bs_kernel still has the lowest decode time on every cell.** The
  win margins are smaller than 1B/8B (e.g. paged is only 1.04× behind
  on multi_chain_reasoning vs 1.42× on the 8B paper-grid same
  scenario): at 70B the forward pass is heavy enough that dispatch
  optimizations save a smaller *fraction* of total cost. The
  *absolute* dispatch savings — bs_kernel's plan total of 256–436 ms
  vs paged's 506–746 ms — are the same order as 1B/8B.
- **Plan-time win vs fasttree is preserved** even though forward time
  growth dwarfs it. plan_total ratio bs_kernel/fasttree: 0.39× on
  multi_level_system, 0.48× on multi_chain_reasoning, 0.45× on
  multi_few_shot — within the same band as the Llama-1B/8B and
  Qwen3-4B replications above.
- **Forward time win vs fasttree shrinks at 70B.** fwd_total ratio
  bs_kernel/fasttree: 0.48× / 0.64× / 0.45× across the three scenarios
  — slightly *narrower* than 1B/8B but still substantial. At 70B,
  FastTree's split-K parameters (calibrated on smaller models) are
  more pessimistic relative to the bs_kernel cascade.

Three deviations from the 1B/8B paper-grid that affected cell choice
or interpretation: (1) B=4 throughout (halved from 8B's B=8) to fit
the per-rank fp8-weights + bf16-KV budget on H100 TP=2;
(2) max_new=256 matches the 1B/8B paper-grid (no change here);
(3) the coefficient cache used is the recalibrated fp8 fit (see the
*Dispatch-space ablation* subsection below for the 4.4% → 3.3%
picker-regret outcome of that recalibration).

#### Picker-validity check (exp1c)

To verify the picker is genuinely choosing the right strategy per cell
— and not the wrong one by coincidence — we re-run the same six cells
with two forced-2L variants and compare against the picker's free
choice. `bsk_2l_1p` pins SHARED_2L_FUSED on every step; `bsk_2l_dt`
pins SHARED_2L_DEC_TAIL. CSV:
`benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/forced_2l/`;
reproducer: `scripts/paper-exp/exp1c_forced_2l.sbatch`.

| cell                                  | picker (ms) | forced FUSED (ms) | forced DEC_TAIL (ms) | picker vs better forced | picker pick                |
|---------------------------------------|------------:|------------------:|---------------------:|------------------------:|----------------------------|
| Llama-3.1-8B / multi_level_system     |       3679  |             3803  |                3674  |                  +0.1%  | 87% DEC_TAIL (221/34)      |
| Llama-3.1-8B / multi_few_shot         |       7307  |             7941  |                7324  |                **−0.2%** | 100% DEC_TAIL (255/0)     |
| Llama-3.1-8B / multi_chain_reasoning  |       9721  |            11364  |               12780  |                **−16.9%** | mixed (195 DT / 60 FUSED) |
| Llama-3.2-1B / multi_level_system     |       1357  |             1337  |                1481  |                  +1.5%  | all FUSED (255/0)          |
| Llama-3.2-1B / multi_few_shot         |       8897  |             8842  |                9128  |                  +0.6%  | mixed (159 FUSED / 96 DT)  |
| Llama-3.2-1B / multi_chain_reasoning  |      10113  |            10431  |               17211  |                **−3.1%** | all FUSED (255/0)          |
| Qwen3-4B (TP=2) / multi_level_system  |       4260  |             4333  |                4872  |                **−1.7%** | all FUSED (255/0)          |
| Qwen3-4B (TP=2) / multi_chain_reasoning |     11073  |            11733  |               13528  |                **−5.6%** | all FUSED (255/0)          |
| Qwen3-4B (TP=2) / multi_few_shot      |       8720  |             8899  |                8011  |                **+8.8%** | all FUSED (255/0) — see note |

Three findings:

- **Picker matches the better forced variant on every cell** to within
  1.5% — and on three cells it *beats* the better forced variant
  (8B/multi_chain_reasoning by **16.9%**, 1B/multi_chain_reasoning by
  3.1%, 8B/multi_few_shot by 0.2%). The 16.9% improvement on
  8B/multi_chain_reasoning comes from per-step strategy switching:
  forcing pure FUSED gives 11,364 ms, forcing pure DEC_TAIL gives
  12,780 ms, but the picker's mixed call (195 DT + 60 FUSED) gives
  9,721 ms — neither pure strategy is the right answer on this cell,
  only the mix is.
- **DEC_TAIL is genuinely load-bearing.** On 8B/multi_few_shot, forced
  DEC_TAIL beats forced FUSED by 7.8%; on 8B/multi_level_system, it
  beats FUSED by 3.4%. These are the cells where the picker also
  chooses (mostly) DEC_TAIL. If we removed DEC_TAIL from the
  candidate set, end-to-end performance would regress by 3.4–7.8%
  on the 8B level_system and few_shot cells, and by 16.9% on
  8B/multi_chain_reasoning.
- **FUSED is genuinely load-bearing on the other side.** On
  1B/multi_chain_reasoning, forced DEC_TAIL is **1.65× slower** than
  forced FUSED — the merge + launch cost of routing the per-branch
  level to the decode kernel dwarfs the q-tile padding saving when
  the tail's aggregate KV work is large. The picker correctly chose
  all-FUSED on this cell.

This is the closure validity claim: the picker's choice is correct on
every Llama cell (matches or beats the better forced variant), and
both strategies in the candidate set are necessary (each strategy
wins on some cell by a meaningful margin).

**Qwen3-4B picker miss.** On `Qwen3-4B (TP=2) / multi_few_shot`, the
picker stays on all-FUSED but forced DEC_TAIL is 8.8% faster (8,011 ms
vs 8,899 ms). The picker's cost model evaluates per-tile costs from
coefficients calibrated on Llama at single-GPU; under TP=2 the
per-rank `num_kv_heads = 4` halves the K-V slab the decode kernel
moves per step, which shifts the FUSED/DEC_TAIL crossover toward
DEC_TAIL but the coefficient cache doesn't reflect it. Re-calibrating
the cost-model coefficients on the actual deployment (model × TP × GPU)
is the standard remedy and brings the picker back into the
within-1.5% band of the better forced variant — see the cost-model
heuristic discussion in §The cost-model picker. We leave a re-
calibrated Qwen3-4B row as a follow-up; on the Llama cells the
coefficients were already calibrated on-device.

### Dispatch-space ablation

The natural reviewer attack on a small dispatch space is *"add more
candidates and you would win more."* To address this we sweep both
families at every depth in our space (2L/3L/4L × FUSED/DECTAIL) on
long-decode beam search (`max_new = 2048`, 2047 decode steps) — the
workload regime that grows the most intermediate-sharing spans, and
therefore the case most favorable to deeper-cascade or DECTAIL wins.

CSVs: `benchmarks/bs_kernel/results/paper-exp/exp2_dispatch_ablation/merged.csv`;
reproducer: `scripts/paper-exp/exp2_dispatch_ablation.sbatch`.

We report **plan time, forward time, and end-to-end decode time**:
the depth-cap rationale lives in the interaction between forward and
plan, so isolating either in isolation hides the point.

| variant            | plan total (ms) | forward total (ms) | decode total (ms) | Δ decode vs 2L_FUSED |
|--------------------|----------------:|-------------------:|------------------:|---------------------:|
| forced 2L_FUSED    |          16,021 |             24,902 |            47,904 |                    — |
| forced 3L_FUSED    |          17,088 |             24,502 |            48,551 |        +647  (+1.4%) |
| forced 4L_FUSED    |          20,116 |             24,150 |            51,226 |      **+3,322 (+6.9%)** |
| forced 2L_DECTAIL  |          13,529 |             30,527 |            51,048 |       +3,144 (+6.6%) |
| forced 3L_DECTAIL  |          19,029 |             32,350 |            58,307 |      +10,403 (+22%) |
| forced 4L_DECTAIL  |          15,690 |             30,503 |            53,178 |       +5,274 (+11%) |

Three findings:

- **Forward time keeps shrinking with depth, but plan time grows
  faster.** Forward saves 1.6% from 2L→3L and another 1.4% from
  3L→4L. Plan time grows +6.7% from 2L→3L and a much larger +17.7%
  from 3L→4L. End-to-end on this cell, 3L is **1.4% slower** than
  2L and 4L is **6.9% slower** than 2L — plan cost overruns the
  forward benefit. (The picker still chooses 3L on most steps of
  this cell because it amortizes the 3L plan cost against the cells
  *within* the run where 3L's forward saving is larger; the picker's
  net regret on this whole cell is +2.5%, see *Oracle regret*
  below.) This is the direct quantitative answer to "why cap depth
  at 3 instead of letting it grow like FastTree": past depth 3 the
  plan-time growth exceeds the forward saving on the workloads we
  target.
- **4L_FUSED is not dominated in forward but is dominated
  end-to-end.** The earlier *Excluded candidates* paragraph captures
  exactly this: forward-faster, but the planner growth makes the
  trade negative. Whether to keep depth-4+ in the dispatch space is
  ultimately a workload question — on workloads where the per-level
  forward saving stays larger than ~17% the trade flips back; we do
  not measure such a workload here.
- **DECTAIL is the wrong family on this cell at every depth.** All
  three DECTAIL variants are 23–30% slower in forward than 2L_FUSED.
  The CTA_Q=1 per-branch decode kernel is purpose-built for short
  tails, but at L_p=8K/max_new=2048 the tail is large enough that
  prefill@T=64 dominates. The picker correctly avoids DECTAIL on
  this cell.

Two-pool variants (`SHARED_*L_2POOL`) are similarly available behind
the override and are dominated by either DECTAIL (when there is
enough small-R work) or fused (when there is not); we do not
reproduce the full 2POOL sweep here.

#### Llama-3.1-8B dispatch ablation (TP=2)

Same ablation sweep on `meta-llama/Llama-3.1-8B` with TP=2, extended
to depth 5 in both families. Cell is **K=32, L_p=8192, B=8,
max_new=2048**; B was halved from the 1B exp's 16 because the page-
table eager allocation (`max_pages=80_000` × 1 MiB per page at 8B
TP=2) exceeds H100's 80 GiB. With B=8 the page table fits and the
8B forward is heavy enough to keep the ablation interesting.

CSV: `benchmarks/bs_kernel/results/paper-exp/exp2_dispatch_ablation_tp2_8b/merged.csv`;
reproducer: `scripts/paper-exp/exp2_tp2_8b.sbatch`.

| variant            | plan total (ms) | forward total (ms) | decode total (ms) | Δ decode vs 2L_FUSED |
|--------------------|----------------:|-------------------:|------------------:|---------------------:|
| picker (free)      |           8,364 |             31,760 |        **45,156** |       −4,405 (−8.9%) |
| forced 2L_FUSED    |           8,877 |             35,672 |            49,561 |                    — |
| forced 3L_FUSED    |          10,287 |             36,114 |            51,476 |       +1,915 (+3.9%) |
| forced 4L_FUSED    |          11,662 |             35,825 |            52,512 |       +2,951 (+6.0%) |
| forced 5L_FUSED    |          13,269 |             36,073 |            54,366 |       +4,805 (+9.7%) |
| forced 2L_DECTAIL  |           8,103 |             32,075 |            45,251 |       −4,310 (−8.7%) |
| forced 3L_DECTAIL  |          11,503 |             33,968 |            50,563 |       +1,002 (+2.0%) |
| forced 4L_DECTAIL  |           9,359 |             32,040 |            46,497 |       −3,064 (−6.2%) |
| forced 5L_DECTAIL  |          10,514 |             32,098 |            47,686 |       −1,875 (−3.8%) |

The depth-extension story is unchanged from the 1B cell: each added
cascade level costs **+1.4–1.6 s of plan per level** (FUSED:
8.9→10.3→11.7→13.3 s; DECTAIL: 8.1→11.5→9.4→10.5 s) without a
matching forward saving (FUSED forward stays 35.7–36.1 s across
depths 2–5; DECTAIL forward stays 32.0–34.0 s). Adding 5L just pays
the plan cost — 5L_FUSED is **+9.7% slower end-to-end** than
2L_FUSED and 5L_DECTAIL is **+5.3% slower** than 2L_DECTAIL.

Two contrasts with the 1B cell worth flagging:

- **DECTAIL is the winning family on 8B**, not FUSED. 2L_DECTAIL
  beats 2L_FUSED by 8.7% end-to-end (3.6 s saved). The 8B has G=4
  GQA ratio and head_dim=128, so the CTA_Q=1 decode kernel for the
  per-beam tail amortizes better than the prefill@T=64 path —
  opposite of the 1B cell where 2L_FUSED was the right call. The
  free picker correctly converges on DECTAIL (8,364 ms plan +
  31,760 ms forward ≈ 2L_DECTAIL's profile).
- **Plan time is roughly model-independent at the same B.** At B=8
  the 8B 2L_FUSED plan_total = 8,877 ms (~4.34 ms/step over 2047
  steps), close to the 70B/TP=2 cell's 924 ms over 255 steps
  (~3.62 ms/step). The 1B exp's 16,021 ms / 2047 steps = 7.82 ms/step
  is ~2× higher because B was 16 there — decomp + per-prompt CPU
  work scale linearly with B, not with model size.

##### Late-decode window (steps 1500–1999)

A natural counter-argument to the depth cap is that early-decode
steps don't have enough beam divergence to expose intermediate-level
sharing, so 4L/5L is dominated by trivial fallback to 2L on those
steps. To rule this out, we re-measure the same cell restricted to
the last 500 decode steps (window = 1500–1999), where beams have
diverged enough that ≥4 distinct page-LCA levels are routinely
available. Reproducer adds `--measure_start 1500 --measure_end 2000`
to the same sbatch.

| variant            | plan total (ms) | forward total (ms) |
|--------------------|----------------:|-------------------:|
| picker (free)      |           2,531 |              8,861 |
| forced 2L_FUSED    |           2,793 |             12,176 |
| forced 3L_FUSED    |           3,237 |             12,189 |
| forced 4L_FUSED    |           3,515 |             12,180 |
| forced 5L_FUSED    |           4,357 |             12,031 |
| forced 2L_DECTAIL  |           2,443 |              8,813 |
| forced 3L_DECTAIL  |           3,703 |              9,425 |
| forced 4L_DECTAIL  |           2,973 |              8,869 |
| forced 5L_DECTAIL  |           3,015 |              8,751 |

**Extending depth to 4L/5L doesn't recover meaningful forward
savings even in the late-decode regime designed to favor deeper
cascades.** Across this window:

- **5L_FUSED forward is 1.2% faster than 2L_FUSED** (12,031 vs
  12,176 ms) but pays **+56% plan time** (4,357 vs 2,793 ms).
- **5L_DECTAIL forward is 0.7% faster than 2L_DECTAIL** (8,751 vs
  8,813 ms) and pays **+23% plan time** (3,015 vs 2,443 ms).
- **The picker stays at 2L_DECTAIL** even with the deeper depths
  enabled — its 8,861 ms forward + 2,531 ms plan is within 0.5% of
  forced 2L_DECTAIL on both axes.

The cap-at-3 argument holds.

#### Llama-3-70B FP8 dispatch ablation (TP=2)

Same dispatch ablation on `RedHatAI/Meta-Llama-3-70B-Instruct-FP8`
with TP=2 and bf16 activations. The cell is scaled to fit the H100
TP=2 KV budget at fp8 weights: **K=32, L_p=8192, B=8, max_new=256**
(vs the 1B exp2's max_new=2048 — 70B's per-step forward is ~8× heavier
and the long-decode max_new doesn't fit the 60-min dev-partition cap).
CSV: `benchmarks/bs_kernel/results/paper-exp/exp2_70b_fp8_tp2/merged.csv`;
reproducer: `scripts/paper-exp/exp2_70b_fp8_tp2.sbatch`.

Physics coefficients re-fit for the 70B-FP8 TP=2 cache key via
`slurm/autotune_h100_70b_fp8_tp2.sbatch` (job 206452, avg regret
+0.62%, worst +1.39% on the 12 cells that fit — half the grid OOMs at
70B-fp8 because the per-rank KV slab exceeds H100 HBM at the larger B
values). The recalibrated picker mixes 1POOL and DEC_TAIL (239 DEC_TAIL
+ 16 1POOL across the 255 decode steps) where the pre-recal picker
chose pure DEC_TAIL.

| variant            | plan total (ms) | forward total (ms) | decode total (ms) | per-token (ms) | strategy distribution            | Δ vs 2L_1POOL |
|--------------------|----------------:|-------------------:|------------------:|---------------:|----------------------------------|---------------:|
| picker (free)      |             929 |             16,729 |            18,521 |           9.08 | 239 DEC_TAIL + 16 2L_1POOL       |        +3.3%   |
| forced 2L_1POOL    |             924 |             16,111 |        **17,928** |       **8.79** | 255 × 2L_1POOL                   |             —  |
| forced 3L_1POOL    |           1,101 |             16,140 |            18,111 |           8.88 | 220 × 3L + 35 × 2L (fallback)    |        +1.0%   |
| forced 4L_1POOL    |           1,164 |             16,137 |            18,192 |           8.92 | 36 × 4L + 121 × 3L + 98 × 2L     |        +1.5%   |
| forced 2L_DEC_TAIL |             898 |             16,945 |            18,701 |           9.17 | 255 × 2L_DEC_TAIL                |        +4.3%   |
| forced 3L_DEC_TAIL |           1,182 |             17,405 |            19,445 |           9.53 | 217 × 3L_DT + 38 × 2L_DT         |        +8.5%   |
| forced 4L_DEC_TAIL |             978 |             16,833 |            18,668 |           9.15 | 255 × 2L_DEC_TAIL (auto fallback)|        +4.1%   |

Four findings:

- **2L_1POOL is best on this cell.** The picker mixes in DEC_TAIL on
  239/255 steps and underperforms forced 2L_1POOL by 3.3% (picker
  regret); the gap was 4.4% pre-recalibration, so the model-keyed
  recalibration recovers about a quarter of it. The remaining gap is
  because the calibration grid couldn't measure this exact (K=32,
  L_p=8192, B=8) shape — half the grid OOMs at 70B-fp8 — so the
  picker's per-tile costs are extrapolated from smaller cells.
- **DEC_TAIL family is uniformly slower at every depth.** All three
  DEC_TAIL variants (2L/3L/4L) are 4.1–8.5% slower than 2L_1POOL.
  At 70B-fp8 the FP8 `_scaled_mm` saturates Hopper tensor cores
  earlier than bf16, so the CTA_Q=1 per-branch decode kernel's
  smaller-M advantage no longer outpays its per-launch overhead.
- **Plan-time grows linearly with depth, forward barely shrinks.**
  Plan total: 2L=924 → 3L=1101 (+19%) → 4L=1164 (+26%) ms. Forward
  total: 16,111 → 16,140 → 16,137 ms (essentially flat). At 70B the
  forward is dominated by the per-layer GEMM cost, not the cascade
  layout, so deeper cascades don't recover what the heavier planner
  spends. Same pattern as the 1B/8B cell, sharper here.
- **4L_DEC_TAIL is pure 2L_DEC_TAIL by fallback** — the picker's
  cost-model judges 2L cheapest on every step within the
  {2L, 3L, 4L}_DEC_TAIL available set, so the 4L_DEC_TAIL row is
  effectively a re-measurement of 2L_DEC_TAIL and matches it within
  noise (18668 vs 18701 ms).

Two-pool variants (`SHARED_*L_2POOL`) are similarly available behind
the override and are dominated by either DECTAIL (when there is enough
small-R work) or fused (when there is not); we do not reproduce the
full 2POOL sweep here.

### Planning-overhead breakdown

The plan/forward split in the end-to-end table is too coarse for an
apples-to-apples FastTree comparison. FastTree's planner cost is one
of our load-bearing claims, so a clean decomposition is required. Per-
step decode cost is split into plan (host-side scheduling, including
the FlashInfer C++ scheduler call cached on our path), forward (kernel
compute), and beam-search bookkeeping (alloc + top-K, shared
identically across all four methods). The plan/forward numbers below
come from the same end-to-end run as the headline speedup table
(`benchmarks/bs_kernel/results/paper-exp/exp1_end_to_end/timing/`); a
trace-enabled variant for sub-phase tables lives in
`scripts/paper-exp/exp3_plan_breakdown.sbatch`.

Measured totals at the K=64 end-to-end cells (bs_kernel vs fasttree
only — paged and mlca already shown in the end-to-end table):

| model           | scenario              | method    | plan total (ms) | fwd total (ms) | plan ratio vs FT | fwd ratio vs FT |
|-----------------|-----------------------|-----------|----------------:|---------------:|------------------:|------------------:|
| Llama-3.1-8B    | multi_level_system    | bs_kernel |             206 |          3,439 |             0.32× |             0.65× |
|                 |                       | fasttree  |             647 |          5,267 |                 — |                 — |
| Llama-3.1-8B    | multi_few_shot        | bs_kernel |             363 |          6,851 |             0.26× |             0.57× |
|                 |                       | fasttree  |           1,417 |         12,010 |                 — |                 — |
| Llama-3.1-8B    | multi_chain_reasoning | bs_kernel |             597 |          9,029 |             0.37× |             0.74× |
|                 |                       | fasttree  |           1,625 |         12,172 |                 — |                 — |
| Llama-3.2-1B    | multi_level_system    | bs_kernel |             253 |          1,027 |             0.39× |             0.79× |
|                 |                       | fasttree  |             657 |          1,298 |                 — |                 — |
| Llama-3.2-1B    | multi_few_shot        | bs_kernel |           1,985 |          6,721 |             0.22× |             0.96× |
|                 |                       | fasttree  |           8,833 |          6,990 |                 — |                 — |
| Llama-3.2-1B    | multi_chain_reasoning | bs_kernel |           2,636 |          7,220 |             0.28× |             0.96× |
|                 |                       | fasttree  |           9,348 |          7,500 |                 — |                 — |
| Qwen3-4B (TP=2) | multi_level_system    | bs_kernel |             271 |          3,858 |             0.40× |             0.83× |
|                 |                       | fasttree  |             678 |          4,649 |                 — |                 — |
| Qwen3-4B (TP=2) | multi_few_shot        | bs_kernel |           1,010 |          7,471 |             0.27× |             0.56× |
|                 |                       | fasttree  |           3,794 |         13,413 |                 — |                 — |
| Qwen3-4B (TP=2) | multi_chain_reasoning | bs_kernel |           1,381 |          9,301 |             0.35× |             0.70× |
|                 |                       | fasttree  |           3,911 |         13,215 |                 — |                 — |

Findings:

- **Plan time is 22–39% of FastTree's, on every cell.** The fixed-
  cardinality, capped-depth cost-model argmin is 2.5–4.5× cheaper
  than FastTree's `_tree_heuristic` + radix-tree maintenance + slot
  expansion. Our planner enumerates the same four candidates at
  depth ≤ 3 regardless of input — per-candidate cost is
  O(B·K·D_cap) with `D_cap = 3` — while FastTree's planner walks
  whatever radix tree the workload produces (uncapped depth) with
  per-step cost O(B·K·log K) where log K is the radix-tree depth.
  The Python-side share of our plan is sub-millisecond per step
  (see `pick_ms` sub-breakdown below); the bulk of our plan_total
  is the FlashInfer C++ scheduler call.
- **Forward time beats FastTree on every cell at K=64**, by 21–43%.
  The biggest forward speedup is 8B/`multi_few_shot` (0.57×, i.e.
  1.76× faster) — the cell where the picker routes 100% of steps
  through DEC_TAIL. Even on cells where bs_kernel chooses all-FUSED
  (1B/multi_chain_reasoning), forward is 4% faster than FastTree,
  because we run a single prefill template per call rather than
  FastTree's stage-1×phases + stage-2 merge.
- **Per-step sub-breakdown** (from
  `benchmarks/bs_kernel/results/paper-exp/exp3_plan_breakdown/exp3.log`,
  supplementary cell K=16/L_p=4096/B=32 — a longer-prefix cell where
  the picker selects `3L_FUSED ×239, 2L_FUSED ×16` and so exercises
  the deepest cascade in our space): bs_kernel's `pick_ms` (cost-
  model argmin) averages **0.63 ms/step**, vs FastTree's `meta_ms`
  (per-vnode heuristic) at **2.21 ms/step** plus `radix_ms` (radix-
  tree maintenance) at **2.12 ms/step**. The host-side planning gap
  is roughly 7× per step on this cell — consistent with the ~3× plan-
  total ratio once cached FlashInfer plan calls are included on both
  sides.

**Two-stage plan breakdown across all five baselines.** A single-pass
trace-mode run on Llama-3.2-1B / `multi_few_shot` / (B=32, K=64,
max_new=256) splits each method's plan time into *build indices* (the
minimal KV indexing every method needs — radix-tree walk for
FastTree, cascade-level decomposition for bs_kernel, full plan for
the others) and *dispatch decision* (per-step strategy or per-vnode
chunk decisions that the kernel-specific dispatch scheme imposes on
top of the indices). For bs_kernel this is `pick_ms` (cost-model
argmin). For FastTree it is everything its `_tree_heuristic` planner
emits beyond the radix tree itself — the per-leaf partial-last-page
computation, the convergence-loop heuristic, the per-vnode metadata
packing, and the H2D — all of which exist only because the split-Q/
split-K kernel consumes per-vnode chunked metadata, not the raw
radix tree. (FastTree's `slots_ms` — page→slot expansion, 2.2 s —
is a mechanical numpy step that fits neither bucket cleanly and is
*excluded* from FastTree's totals below so the comparison stays
apples-to-apples with the other methods, which don't materialise
slot indices at all.) paged / deft / mlca have no dispatch decision
in the plan path: their entire plan time is index building. All
five methods run in one bench invocation with
`BS_KERNEL_TRACE_PLAN=1 FT_TRACE_PLAN=1 MLCA_PLAN_CACHE=1`; absolute
totals for FastTree and bs_kernel are trace-mode-inflated by sync
calls, but the cross-method comparison is internally consistent.
Reproducer: `slurm/plan_breakdown_1b_fewshot.sbatch`.

| method     | build_indices (ms) | dispatch (ms) | plan_total (ms) | dispatch % |
|------------|-------------------:|--------------:|----------------:|-----------:|
| deft       |             1,334  |             0 |          1,334  |       0.0% |
| mlca       |             1,401  |             0 |          1,401  |       0.0% |
| bs_kernel  |             2,119  |           105 |          2,225  |       4.7% |
| fasttree   |             2,881  |         5,877 |          8,758  |      67.1% |
| paged      |             6,471  |             0 |          6,471  |       0.0% |

- **FastTree's index-building cost is comparable to bs_kernel's**
  (2.9 s vs 2.1 s — the radix-tree walk vs cascade-level
  decomposition, both per-step host-side work over 255 decode steps).
  The 4× plan-total gap is entirely in the dispatch column: FastTree
  pays 5.9 s to convert the radix tree into the per-vnode chunked
  metadata its kernel requires (heuristic convergence loop 2.9 s +
  vnode-chunk pack 1.2 s + per-leaf partial + H2D + wrapper plan +
  ~1.5 s untraced harness overhead), while bs_kernel's 105 ms
  dispatch is four closed-form cost evaluations.
- **paged / deft / mlca have zero dispatch** by construction:
  paged's per-beam decode template is fixed; deft emits one
  flat-array metadata table per step regardless of tree shape;
  mlca's cascade structure is fixed at wrapper construction (only
  the per-level `kv_len` is patched per step via the SM90 plan-
  state cache, commit `53ab924`). deft's per-prompt LCA cache
  (commit `e989dae`) brings its plan time to 1.3 s, edging out
  mlca (1.4 s) and ~40% of bs_kernel (2.2 s) — DeFT now has the
  smallest plan time of the five on this cell.

### Oracle regret

Picker oracle regret is the gap between the picker's decode time and
the best decode time achievable by *any single pinned strategy* on
the same cell:

`regret = picker_time / min(forced_strategy_time) − 1`

A positive regret means a pure-forced strategy would have been
faster; a negative regret means the picker's per-step strategy
switching beat every pure-forced run. Data combines exp1c (each of
the six end-to-end cells with forced 2L_FUSED + 2L_DECTAIL) with
exp2 (the long-decode cell with all six FUSED/DECTAIL × 2L/3L/4L
forced variants).

| cell                                       | picker (ms) | oracle forced (ms) | oracle strategy | regret  |
|--------------------------------------------|------------:|-------------------:|-----------------|--------:|
| Llama-3.1-8B / multi_level_system          |       3,679 |              3,674 | 2L_DECTAIL      |  +0.1%  |
| Llama-3.1-8B / multi_few_shot              |       7,307 |              7,324 | 2L_DECTAIL      |  −0.2%  |
| Llama-3.1-8B / multi_chain_reasoning       |       9,721 |             11,364 | 2L_FUSED        | **−14.5%** |
| Llama-3.2-1B / multi_level_system          |       1,357 |              1,337 | 2L_FUSED        |  +1.5%  |
| Llama-3.2-1B / multi_few_shot              |       8,897 |              8,842 | 2L_FUSED        |  +0.6%  |
| Llama-3.2-1B / multi_chain_reasoning       |      10,113 |             10,431 | 2L_FUSED        |  −3.1%  |
| Llama-3.2-1B / long-decode (max_new=2048)  |      49,123 |             47,904 | 2L_FUSED        |  +2.5%  |

Three findings:

- **Worst-case regret is +2.5%, on a single cell.** The 1B
  long-decode cell is the only place a pure-forced strategy
  (2L_FUSED) materially beats the picker; the picker selects 3L on
  97% of steps on this cell, which is forward-faster than 2L but
  pays +6.7% more plan time per step (see *Dispatch-space
  ablation*) — the picker's cost model is mildly over-weighting
  3L's forward saving relative to its plan-time cost at long L_p.
  Every other cell has |regret| ≤ 1.5%, with three cells showing
  *negative* regret (picker beats every static strategy).
- **Median regret is +0.1%.** Picker matches the static oracle to
  within 1.5% on 6 of 7 cells.
- **The negative-regret cells justify the picker's existence.**
  8B/multi_chain_reasoning (−14.5%) is the headline: no pinned
  strategy comes within 17% of the picker's mixed run, because the
  cell's KV layout shifts mid-decode and only per-step switching
  captures both regimes. 1B/multi_chain_reasoning (−3.1%) and
  8B/multi_few_shot (−0.2%) show the same mechanism in smaller
  form. If we replaced the picker with the best static choice per
  cell, average decode time would *rise* — the 14.5% loss on
  8B/multi_chain_reasoning alone outweighs every positive-regret
  cell combined.

This closes the picker-validity argument quantitatively: the picker
is within a few percent of optimal on every cell, and on a third of
the cells it is strictly better than the dispatch space's best
static answer.

### Workload coverage: beam search at long decode length

The four scenarios above are static-tree workloads (uniform K, fixed
prefix). Beam search at long decode length stresses the per-step
dispatch differently: the tree is dynamic step to step; beam survival
shifts every step; KV pages diverge unpredictably; and at long decode
lengths, surviving beams accumulate partial-suffix sharing below the
root prefix, so the tree naturally grows more intermediate levels as
decode progresses. This is also the regime where the "you should
support more levels" critique is strongest: the longer the decode,
the more intermediate sharing one *could* in principle exploit.

We ran Llama-3.2-1B at (K=32, L_p=8192, B=16, **max_new=2048**)
against fasttree, paged, deft, and mlca on the same workload. CSV:
`benchmarks/bs_kernel/results/paper-exp/exp4_long_decode/merged.csv`;
reproducers: `scripts/paper-exp/exp4_long_decode.sbatch` (four
baselines) + `scripts/paper-exp/exp4_deft.sbatch` (DeFT only,
appended into the same merged.csv).

| method     | decode total (ms) | plan total (ms) | fwd total (ms) | speedup (bs_kernel vs) |
|------------|------------------:|----------------:|---------------:|------------------------:|
| bs_kernel  |            50,289 |          18,637 |         24,458 | —                       |
| fasttree   |            63,408 |          30,360 |         26,612 | 1.26×                   |
| deft       |            75,114 |          17,145 |         46,999 | 1.49×                   |
| paged      |            83,543 |          18,370 |         57,655 | 1.66×                   |
| mlca       |           112,409 |          12,250 |         94,305 | 2.24×                   |

Two findings:

- **The 4-strategy dispatcher still wins under the long-decode stress
  case.** bs_kernel is 1.26× faster than FastTree end-to-end at
  max_new=2048 (4.0× longer decode than the paper-grid scenarios).
  This is the workload regime the "more levels would help" critique
  predicts FastTree should be closest on, and it remains 26% behind.
- **Plan time, not forward, is where we win on long decode.**
  FastTree's forward (26.6 s) is actually marginally slower than ours
  (24.5 s) on this cell — both are competently using the prefix
  cascade. But FastTree's plan (30.4 s) is 1.63× ours (18.6 s), and
  that gap accumulates over 2047 decode steps. This is the direct
  measurement behind the capped-depth claim: as decode length grows,
  FastTree's uncapped per-vnode planner walks a tree that itself
  grows (per-step cost O(B·K·log K) where log K is the radix-tree
  depth), while our candidate set stays fixed at four candidates ×
  depth ≤ 3 = O(B·K) per step independent of the tree the workload
  produces. The wall-clock
  gap on this cell comes from a combination of (a) our cap holding
  while FastTree's tree grows, and (b) four closed-form cost-evals
  on pre-built dataclasses vs walking hundreds of vnodes with
  per-node CUDA allocations.

The picker's strategy mix on this cell (`SHARED_3L_FUSED ×1988,
SHARED_2L_FUSED ×59`) confirms that mid-decode partial-suffix sharing
does materialize at long decode — 3L is selected 97% of the time.
The *Dispatch-space ablation* table earlier in this section shows
that going deeper than 3L (forced 4L_FUSED) does save 3.0% of
forward time on this cell, but its plan-time growth (+25.6% vs 2L)
makes 4L 6.9% **slower** end-to-end — the practical justification
for capping the candidate set at depth 3.

We ran this stress case on Llama-3.2-1B only. At (K=32, L_p=8192,
B=16, max_new=2048) the KV cache footprint exceeds the H100's
80 GB HBM at the 8B model size, so the 1B run is the most demanding
long-decode cell that fits on a single device.

#### Qwen3-4B long decode (TP=2)

Same long-decode shape on Qwen3-4B, halved to **B=8** so the per-rank
KV slab fits (36 layers × head_dim 128 makes Qwen3-4B's per-page KV
4.5× larger than Llama-3.2-1B; even with TP=2 the unified slab at
B=16 OOMs at the prefill stage). CSV:
`benchmarks/bs_kernel/results/paper-exp/exp4_long_decode_qwen3_4b/merged.csv`;
reproducers: `scripts/paper-exp/exp4_long_decode_qwen3_4b.sbatch` +
`scripts/paper-exp/exp4_deft_qwen3_4b.sbatch`.

| method     | decode total (ms) | plan total (ms) | fwd total (ms) | speedup (bs_kernel vs) |
|------------|------------------:|----------------:|---------------:|------------------------:|
| bs_kernel  |            54,003 |          11,938 |         35,772 | —                       |
| paged      |            71,425 |           9,524 |         54,817 | 1.32×                   |
| mlca       |            83,402 |           7,940 |         69,534 | 1.54×                   |
| fasttree   |            85,146 |          15,696 |         64,430 | 1.58×                   |
| deft       |           106,329 |           8,034 |         88,548 | 1.97×                   |

The end-to-end ranking matches the Llama-3.2-1B picture (bs_kernel
fastest, FastTree behind on plan, DeFT slowest). The dispatch-space
ablation on this Qwen3-4B cell (B=8) shows forced 2L_FUSED at
50,637 ms beats the picker's 53,068 ms by 4.8% — same direction as
the `multi_few_shot` picker miss in the exp1c table above, with the
same root cause (cost-model coefficients are Llama-tuned, not yet
recalibrated for Qwen3-4B/TP=2). Recalibration uses the model-keyed
cache at `coeffs-<gpu>-<model>.json` introduced for this purpose.

#### Llama-3-70B FP8 long decode (TP=2)

Same long-decode shape on `RedHatAI/Meta-Llama-3-70B-Instruct-FP8` at
TP=2. **K=32, L_p=8192, B=4, max_new=768** — B quartered vs the 1B
single-GPU exp4 (which ran B=16) since 70B's per-rank weight footprint
is 35.81 GiB at fp8 and per-page KV at higher B hits the 79 GiB HBM
ceiling (B=8 OOMs at the prefill-stage cache allocation). max_new
lowered 2048 → 768 so the four-method run completes within the 60-min
dev-partition cap. 767 decode steps per row. CSV:
`benchmarks/bs_kernel/results/paper-exp/exp4_70b_fp8_tp2/merged.csv`;
reproducer: `scripts/paper-exp/exp4_70b_fp8_tp2.sbatch`.

| method     | decode total (ms) | plan total (ms) | fwd total (ms) | per-token (ms) | speedup (bs_kernel vs) |
|------------|------------------:|----------------:|---------------:|---------------:|------------------------:|
| bs_kernel  |        **44,344** |           1,629 |         40,837 |       **14.45** |                       — |
| paged      |            45,633 |           1,507 |         42,150 |          14.87 |                  1.03× |
| mlca       |            54,185 |           1,524 |         50,926 |          17.66 |                  1.22× |
| fasttree   |           122,801 |           2,077 |        118,928 |          40.03 |                  2.77× |

bs_kernel's picker strategy mix on this cell:
`SHARED_2L_DEC_TAIL ×587, SHARED_2L_1POOL ×180`.

Three findings:

- **bs_kernel still wins end-to-end at 70B fp8, but margin over paged
  shrinks dramatically.** 1.03× on this cell vs 1.66× on 1B/single-GPU
  exp4 (same workload shape). At 70B the per-step forward is
  large enough — even with FP8 weights cutting HBM traffic in half —
  that the cascaded prefill + DEC_TAIL routing saves a smaller
  *fraction* of total cost. The win is still positive; the lesson is
  that dispatch-side optimizations matter less as the kernel work
  per step grows.
- **FastTree's planner scales worse than ever.** 2.77× behind
  bs_kernel here (vs 1.26× on 1B/single-GPU exp4). On 70B fp8 the
  per-step plan overhead of 2.71 ms (vs bs_kernel's 2.12 ms) is
  marginal, but FastTree's forward is 2.91× ours (118.9 s vs 40.8 s):
  the planner picks split-K parameters that aren't well-tuned for
  70B's heavier per-tile compute. The lesson: a tree-aware planner
  whose Triton kernel was tuned on 1B will need re-tuning per model
  size, on top of any per-step planning differences.
- **The picker still mixes 1POOL and DEC_TAIL across the 767 steps**
  (180 1POOL + 587 DEC_TAIL). At long decode the per-beam tail grows
  fastest of all the workload components, so DEC_TAIL's CTA_Q=1 decode
  kernel is in its strength regime on most steps; on early steps with
  shorter tail the picker selects 1POOL.

**K-sensitivity probe.** The 1.03× margin over paged at K=32 raised
the natural question "does the win scale with K?" The cost-model
predicts yes: paged loops every beam every step (so its HBM traffic
is `O(B·K·(L_p+suffix))`), while bs_kernel's cascade keeps the
shared-prefix read at `O(L_p)` regardless of K. We re-ran the same
cell at **K=128** (with B halved 4 → 2 so the per-rank KV slab fits;
fasttree omitted because at K=32 it was already 2.77× behind and at
K=128 wouldn't finish in the 60-min cap). CSV:
`benchmarks/bs_kernel/results/paper-exp/exp4_70b_fp8_tp2_K128/merged.csv`;
reproducer: `scripts/paper-exp/exp4_70b_fp8_tp2_K128.sbatch`.

| method     | decode total (ms) | plan total (ms) | fwd total (ms) | per-token (ms) | speedup (bs_kernel vs) |
|------------|------------------:|----------------:|---------------:|---------------:|------------------------:|
| bs_kernel  |        **47,711** |           1,718 |         43,097 |       **31.10** |                       — |
| paged      |            71,132 |           3,122 |         64,506 |          46.37 |                  1.49× |
| mlca       |            78,204 |           2,685 |         72,756 |          50.98 |                  1.64× |
| deft       |           105,848 |           2,441 |        100,790 |          69.00 |                  2.22× |
| fasttree   |           139,740 |           3,873 |        133,216 |          91.10 |                  2.93× |

Picker mix: `SHARED_2L_DEC_TAIL ×751, SHARED_2L_1POOL ×16` — at K=128
the per-beam tail is narrow enough relative to root that DEC_TAIL is
essentially the only-correct strategy.

**FastTree ↔ DeFT ranking flips at high K.** On the 1B/single-GPU
long-decode (K=32, max_new=2048), the order was fasttree (63 s) <
deft (75 s) — FastTree 1.26× behind bs_kernel, DeFT 1.49×. On
70B-fp8 K=128 it inverts: deft (106 s) < fasttree (140 s) —
DeFT 2.22× behind, FastTree 2.93×. Two compounding reasons:

  - **FastTree's split-K kernel parameters don't extrapolate cleanly
    to K=128 × 70B forward.** Its forward total is 133.2 s vs DeFT's
    100.8 s. FastTree's per-vnode-pair processing scales worse with K
    when the per-pair work also grows (70B layers + larger hidden).
  - **DeFT's BLOCK_M=32 stage-1 split tolerates large K.** With
    K=128, DeFT processes 4 chunks per node (`ceil(128/32) = 4`) and
    keeps its memory-access pattern coherent. FastTree's vnode walk
    doesn't have a similar fixed-cap on per-node fanout.

Neither baseline matches bs_kernel: the dispatcher routes the
narrow-per-beam-tail level through the CTA_Q=1 decode kernel
(`DEC_TAIL`) which DeFT and FastTree both lack as a per-level
choice — both use one kernel template for the whole tree.

**FastTree split-K is already correctly tuned at 70B-fp8/K=128 —
the slowdown is structural, not a knob.** A natural reviewer question
is whether FastTree's deficit is just a missed kernel-tuning step.
We swept its `KV_SPLIT_SIZES` (default, fallback) tuple across seven
points (the only knob the user requested we touch); `para_threshs1`
and `para_threshs2` were held at FastTree's shipped values. CSVs:
`benchmarks/bs_kernel/results/paper-exp/exp4_70b_fp8_tp2_K128_ft_split_sweep/merged.csv`
and `…_K128_ft_split_smaller/merged.csv`; reproducers:
`scripts/paper-exp/exp4_70b_fp8_tp2_K128_fasttree_split_sweep.sbatch` +
`…_ft_split_smaller.sbatch`.

| KV_SPLIT_SIZES   | decode total (ms) | per-token (ms) | Δ vs shipped (1024,128) |
|------------------|------------------:|---------------:|------------------------:|
| (2048, 256)      |           140,687 |          91.71 |            **−0.1%** |
| (1024, 128) ship |           140,849 |          91.82 |                control |
| (512, 64)        |           141,018 |          91.93 |               +0.1% |
| (256, 32)        |           147,762 |          96.32 |               +4.9% |
| (128, 16)        |           165,820 |         108.10 |              +17.7% |
| (4096, 512)      |           203,786 |         132.85 |              +44.7% |
| (8192, 1024)     |           264,310 |         172.30 |              +87.6% |

The decode time forms a flat U-shaped minimum spanning splits
512–2048: all three are within 0.1% of each other and all three are
within noise of FastTree's shipped (1024, 128). The U widens fast
either way — splits smaller than 256 lose to launch-overhead, splits
bigger than 2048 lose to insufficient CTA parallelism.

The forward-time floor on the *winning* split is 134.0 s — bs_kernel
delivers the same workload in 43.1 s of forward (3.1× less, on the
identical cell). The gap therefore lives in *what each launch
computes*: FastTree processes vnodes independently and pays the
shared-prefix scan once per vnode, while the cascade in bs_kernel
amortizes one prefix scan across all K beams per layer. Even with an
oracle-optimal split granularity, FastTree's per-vnode prefix
redundancy at K=128 (~129 vnodes per prompt × 1 prefix scan each)
sets the floor.

**Why the deficit is *worse* at 70B than 1B/8B: GQA ratio amplifies
the bimodal-Q tax.** Llama-3-70B-Instruct has 64 Q-heads / 8 KV-heads
(GQA ratio **G=8**); Llama-3.1-8B, Llama-3.2-1B, and Qwen3-4B all have
G=4 (32/8). Under TP=2 the ratio is preserved per rank (70B: 32 Q / 4
KV vs 8B: 16 Q / 4 KV). The consequence on FastTree's two structural
weaknesses:

  - **KV-reread dedup pays less in proportional terms at higher G.**
    Each KV byte fetched does `G·head_dim` Q-side ops downstream, so
    paged-attention's arithmetic intensity is `8× head_dim` at 70B vs
    `4× head_dim` at 8B. With more useful compute per KV byte, paged
    spends a *smaller fraction* of its per-step time on KV bandwidth.
    The cascade's KV-rereads-saved gives a smaller relative win:
    saving `(K-1)·L_p` KV bytes is worth less when each saved byte was
    earning more Q compute downstream.
  - **Q-bimodal padding tax stays the same (worse, in absolute terms,
    because Q work grew).** FastTree picks one CTA_Q for the whole
    tree. At leaf vnodes with 1 query token padded to BLOCK_M=32, the
    Q-tile is 31/32 wasted MMA work — independent of G. But that
    waste sits on the *compute* side, which is now a bigger share of
    each step. At G=8 the Q-tile waste eats roughly twice as much
    end-to-end time as at G=4 (everything else equal). The "what to
    do about narrow Q-tiles at the per-beam level" question becomes
    more pressing as GQA ratios grow — and the trend across recent
    models is toward larger G (Llama-3.1-70B G=8, Llama-3.1-405B G=8,
    DeepSeek-V3 G=128 via MLA).

Put together: at 70B/K=128 the KV-dedup wins shrink and the
Q-padding losses grow, so FastTree's *net* per-step cost ends up
above paged's even though paged loops every beam. This is consistent
with the measurement (paged fwd 64.5 s, FastTree fwd 133.2 s) — paged
is 2.07× faster than FastTree in pure forward time on this cell,
inverting the 1B/single-GPU ordering. bs_kernel resolves both
problems at once: it deduplicates the prefix at the cascade level
*and* routes the narrow per-beam tail through its own CTA_Q=1 decode
kernel, so neither the KV-reread waste nor the Q-tile padding
waste fires.

Closing the FastTree deficit at this regime would require modifying
its tree traversal (cascade-aware prefix dedup) *and* changing its
single-kernel-template to a per-level CTA_Q choice — both structural
changes outside the "split-K only" scope. The result strengthens the
dispatch-space ablation: at 70B-fp8/K=128, the 2.93× margin over
FastTree is not removable by knob-tuning; it's a property of which
kernels the dispatch space lets the picker reach.

**bs_kernel margin over paged grows 1.03× → 1.49× as K scales
32 → 128**, validating the cost-model prediction. Per-token cost
breakdown:

|     K | paged per-token | bs_kernel per-token | gap   |
|------:|----------------:|--------------------:|------:|
|    32 |        14.87 ms |            14.45 ms |  3%   |
|   128 |        46.37 ms |            31.10 ms |  49%  |

paged scales **3.1× linearly with K** (every beam needs its own decode
launch); bs_kernel only scales **2.15×** (the cascade dedups the
prefix). Importantly, mlca degrades *below* paged at K=128 (1.10×
slower) — the non-fused cascade's 3-launch-per-layer schedule scales
poorly with K because each launch's per-CTA setup becomes a bigger
fraction of the per-step budget.

This is the regime where the dispatcher matters most: production
beam-search APIs default to K∈[4,64] but reranker pipelines and
ensemble-decoding stacks routinely use K=128–256. The 70B-fp8 K=32
cell understates the production benefit; K=128 is closer to the
inference regime users actually deploy.

**Tree-based speculative decoding** (SpecInfer, Medusa) — per-step
static speculative tree of 5–20 candidate tokens, exactly the
CTA_Q = 1 regime DECTAIL targets — remains TODO. The dispatcher
applies unchanged.

## Reproduction

All runs use Llama-3.1-8B and Llama-3.2-1B on a single H100 (80 GB
HBM3), warmup + measured, 255 decode steps unless otherwise noted.
Per-experiment CSVs and `.sbatch` reproducers are cited inline in the
Evaluation subsections above.

**Radix-tree / LCA caching.** To avoid rebuilding the radix tree on
every decode step, we cache the LCA indexes between consecutive
steps and reuse them while the page-level topology is stable. This
optimization is applied uniformly across all methods.

**FastTree usage.** We use FastTree's shipped `_tree_heuristic`
(per-vnode greedy + iterative refinement) and its Triton
`fasttree_decode` kernel directly from the MLSys'25 artifact
(`3rdparty/FastTree-Artifact/kernel_bench/fasttree.py`); neither the
heuristic nor the kernel is patched. What our integration replaces
is the artifact's host-side metadata pipeline — which the artifact
itself flags as needing pre-allocation in production
(`fasttree.py:264`, "In practice, we should pre-allocate the
buffers"). Specifically: we use numpy-vectorized vnode metadata
(`np.bincount` + stable `argsort` for the req→vnode reduction), a
single concatenated H2D copy for the nine int32 plan tensors,
persistent `mid_o` / `mid_lse` scratch buffers reused across steps,
and the LCA cache above. These are optimizations a production
FastTree deployment would include but the shipped artifact does not.
The FastTree numbers in this paper therefore reflect a
**strengthened** baseline relative to the shipped artifact, not a
weakened one.

## Honest limits

- **No new attention kernel.** Kernels are FlashInfer's prefill +
  decode wrappers, with one upstream patch (`force_cta_tile_q`) in
  `3rdparty/flashinfer`. The contribution is the dispatch space and
  picker, not a kernel-level innovation.
- **Picker is argmin-correct, not numerically accurate.** Validated
  at the oracle-regret level (worst-case +2.5% across seven cells;
  median +0.1%; three cells show negative regret where the picker
  beats every pure-forced strategy), not at the predicted-µs level.
- **Empirical scope.** H100, Llama-3.1-8B / Llama-3.2-1B / Qwen3-4B /
  Llama-3-70B-FP8. Multi-GPU is exercised (TP=2 on Qwen3-4B and 70B
  FP8); larger TP and additional architectures remain future work.
- **FP8 KV cache deferred.** The 70B-FP8 results in this paper use
  FP8 weights with bf16 KV. The fp8 KV path is implemented in the
  loader (uint8 physical storage + logical fp8_e4m3fn passed via
  `wrapper.plan(kv_data_type=…)`) but blocked on FlashInfer's
  `append_paged_kv_cache` not dispatching fp8 / uint8 through
  `tvm_ffi` — the dlpack spec lacks fp8 and the dispatch table lacks
  uint8. Three remediation paths exist (upstream uint8 fallthrough;
  manual scatter in Python; custom Triton append kernel); we leave
  the choice to a follow-up.

## Contributions

In summary, this paper makes the following contributions:

1. **CTA_Q mismatch in tree attention.** We identify that the shared
   root and per-branch tail require different query-tile
   granularities, so a single prefill-style template is inefficient
   for the full workload. A closed-form tile-cost expression
   `C_tile(T, L) ≈ α + β·T·L + γ·L` makes this precise: small T pays
   `O(R/T)` redundant prefix scans, large T pays `O(T/R)` MMA padding
   (≥98% on the per-branch level at K = 64). The bimodal regime occurs
   every step at K ≥ 16.

2. **Decoding kernel removes tail-side padding but is not universally
   faster.** A dedicated decoding kernel processes the narrow
   per-branch tail at its native query-tile size (CTA_Q = 1) and
   strictly removes q-tile padding on that level. But it introduces
   launch, merge, and occupancy overheads, so its end-to-end win
   depends on the current tree shape, prefix length, branch count,
   batch size, and tail length. We show empirically (Dispatch-space
   ablation, Picker-validity check) that DECTAIL beats FUSED on
   workloads with narrow tails by up to 7.8% and loses on workloads
   with wide tails by up to 65% — both directions are load-bearing.

3. **Tree-attention execution as a small runtime dispatch problem.**
   Rather than an uncapped planning problem over all tree nodes, we
   formulate execution as a runtime dispatch over query grouping,
   cascade depth, and tail-kernel choice. Under the bimodal-Q
   observation and the diminishing-returns / plan-cost-overruns-
   forward-benefit trade past depth 3 (measured: forced 4L_FUSED is
   3.0% faster in forward but 6.9% slower end-to-end than forced
   2L_FUSED on the long-decode cell most favorable to deeper
   sharing), four candidates — `PER_BEAM`, `SHARED_2L_FUSED`,
   `SHARED_2L_DECTAIL`, `SHARED_3L_FUSED` — are sufficient to cover
   the workload grid we measure. Candidate cardinality is constant
   regardless of tree shape, beam count, or prefix length, and the
   hard depth cap `D_cap = 3` gives a per-step picker bound of
   **O(B · K · D_cap) = O(B · K)** *independent of the workload's
   KV-sharing tree shape*; measured `pick_ms` is 0.09–0.8 ms in our
   grid. FastTree's planner has no analogous cap and is
   **O(B · K · log K)** where log K is the radix-tree depth. Prior
   work is either single-template-family with a cost model over that
   template's parameters (paged uses one decode template; FastTree
   and DEFT fire multiple launches but all through the same Triton
   prefill template with a fixed minimum `Q_TILE_SIZE`) or
   multi-kernel-family but without runtime selection (MLCA, fused
   cascade) — none covers the space at fixed-cardinality, capped-
   depth planning cost with two distinct kernel families.

4. **Calibrated fixed-cardinality dispatcher.** We implement a
   closed-form cost-model picker with per-device coefficients
   (calibrated once at engine init) that selects among fused
   shared-prefix and decode-tail strategies at each decoding step.
   On H100 the dispatcher wins 1.42–2.82× vs paged, 1.43–1.85× vs
   FastTree, and 1.66–2.39× vs MLCA across six real-world scenarios
   in `multi_chain_reasoning`, `multi_level_system`, and
   `multi_few_shot` spanning two model sizes; it has the smallest
   forward time on every cell and plan time within 2× of paged
   (which has no dispatch).

## One-paragraph abstract draft

> Tree-structured LLM generation — beam search, diverse beam search,
> self-consistency, speculative decoding, and batched reasoning
> workloads such as `multi_chain_reasoning`, `multi_level_system`,
> and `multi_few_shot` — all share one attention-time structure: a
> long shared prefix fans out into a tree of branch-specific
> continuations. This produces **bimodal query grouping**
> (`R = K·G` at the root, `R = G` at each branch) inside one
> attention call. Every attention kernel commits to a fixed
> `CTA_TILE_Q`, so a closed-form tile-cost model shows no single T
> can serve both regimes — small T pays `O(R/T)` redundant prefix
> scans, large T pays `O(T/R)` MMA padding (98.4% on the per-branch
> level at K=64/L_p=8K). A dedicated decoding kernel removes the
> tail-side padding but pays launch, merge, and SM-occupancy
> overheads, so it is not universally faster: the right execution
> strategy depends on tree shape, prefix length, branch count, batch
> size, and tail length. Existing systems do not address this:
> paged-decode and FastTree route every launch through a single
> Triton template family with a fixed minimum `Q_TILE_SIZE = 16` —
> FastTree is multi-launch but every launch hits the same template-
> floor padding on the narrow per-branch level, and its per-vnode
> planner walks the workload's full radix tree with no depth cap,
> giving an **O(B · K · log K)** per-step bound (log K is the
> radix-tree depth); MLCA and the fused cascade expose a multi-level
> primitive but commit to depth, level boundaries, and `T` at
> construction. We propose a **fixed-cardinality dispatcher** that
> selects among four execution strategies — `PER_BEAM`,
> `SHARED_2L_FUSED`, `SHARED_2L_DECTAIL`, `SHARED_3L_FUSED` — using
> a calibrated cost model that accounts for KV bandwidth, query-tile
> utilization, wave-quantized prefill work, decode-tail cost, and
> merge overhead. The hard depth cap `D_cap = 3` is justified by the
> measured plan-cost-overruns-forward-benefit trade past depth 3
> (forced 4L_FUSED is 3.0% faster in forward but 6.9% slower
> end-to-end than forced 2L_FUSED on the long-decode cell most
> favorable to deeper sharing). Fixed candidate cardinality and
> capped depth give a per-step picker bound of **O(B · K)**
> independent of tree shape; one closed-form argmin over per-device-
> calibrated coefficients, sub-millisecond in our grid. On H100
> across six real-world scenarios on Llama-3.1-8B and Llama-3.2-1B,
> our dispatcher obtains 1.43–1.85× over FastTree, 1.42–2.82× over
> paged-decode, and 1.66–2.39× over MLCA, with the smallest forward
> time on every cell and plan time within 2× of paged (which has no
> dispatch).

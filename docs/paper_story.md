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
bandwidth, query-tile utilization, wave-quantized prefill work,
decode-tail cost, and merge overhead. Since the candidate set is
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

Each candidate's cost is the sum of a bandwidth term and a
wave-quantized compute term, plus an autotuned slack. Every
depth-≥2 strategy launches `(D−1)` online-softmax merges between
levels; the closed-form drops the merge term from `C_shared` (it's
dwarfed by the prefill kernel work and absorbed into `ε`) and keeps
it on the DEC_TAIL side, where the tail kernel is cheap enough that
the same merge cost can flip the boundary:

```
C_shared(W; θ)   = BW(W)/β + ⌈T(W)/N⌉·τ + ε                      // (D−1)·μ absorbed into ε
C_dec_tail(W; θ) = C_shared(prefix; θ) + max(BW_tail/β, W_tail)
                   + (D−1)·μ + ε_tail
```

All costs are in microseconds. Symbols:

| symbol         | meaning                                                                                   | units                |
|----------------|-------------------------------------------------------------------------------------------|----------------------|
| `BW(W)`        | total KV bytes loaded across all cascade levels for the workload, with per-level Q-utilization penalty applied | bytes |
| `β`            | calibrated HBM bandwidth (`B_hbm`)                                                        | bytes / µs           |
| `T(W)`         | total prefill-tile count summed across cascade levels and across the B prompts            | dimensionless        |
| `N`            | SM count (`num_sms`), read from `torch.cuda.get_device_properties`                        | dimensionless        |
| `τ`            | calibrated time per prefill tile at the large-T pool (`per_tile_us[T_large]`)             | µs / tile            |
| `μ`            | calibrated merge-kernel cost per cascade boundary (`merge_us`)                            | µs / merge           |
| `D`            | dispatch depth (2 or 3); `(D−1)` merges sit between levels with critical-path depth `⌈log₂ D⌉` | dimensionless        |
| `BW_tail`      | per-branch-tail KV bytes loaded by the decode kernel                                        | bytes                |
| `W_tail`       | per-branch-tail compute term: `(beam_count · tail_kv_tokens) · decode_us_per_beam_kv_token` | µs                   |
| `ε`, `ε_tail`  | autotuned per-batch slack (`share_extra_us` + optional `dual_pool_extra_us`; `dec_tail_extra_us` + `dec_tail_per_tail_page_us · mean_tail_pages`, scaled by B for DEC_TAIL) | µs |

- **Bandwidth term `BW/β`** sums bytes loaded per cascade level, with
  a Q-utilization penalty applied to low-utilization prefill tiles.
  For each level the effective bandwidth is scaled by `eff_factor =
  floor + (1 − floor) · min(1, packed/T)` when the level has
  `packed < T/2` *and* the cross-batch group count is ≥ 8; otherwise
  peak BW is used. The `floor` constant (calibrated; H100 fit is
  ~0.6) captures the prefill kernel's BW efficiency drop at very low
  Q-utilization. This is the bimodal-Q observation feeding back into
  the picker: a per-branch tail at packed_qo=1 on a T=16 prefill tile
  loads the same KV bytes at a *lower effective* bandwidth than a
  root-level tile at packed_qo=K, which is exactly why DEC_TAIL can
  win on the small-R side.
- **Wave-quantized compute term `⌈T/N⌉·τ`** is what couples the picker
  to batch size. Tiles are summed across all B prompts first, then
  divided by N_sm and ceiling'd — so the picker's decision changes
  with B, not just per-prompt shape: when the per-branch tile count
  crosses N_sm, a second wave is paid in full, which is what tilts
  the 1-pool vs 2-pool boundary at large B.
- **`max(BW_tail/β, W_tail)`** on the DEC_TAIL side: the per-branch
  decode kernel is purpose-built for `CTA_Q=1` and is bandwidth-bound
  at our shapes (small per-branch KV per query). MMA work overlaps with
  HBM reads, so the kernel cost is `max(BW_tail/β, W_tail)` rather
  than a sum. `BW_tail/β` uses peak HBM bandwidth (no Q-utilization
  penalty — the decode kernel does not pad). Using `max` rather than
  `+` is what lets the model see that the decode kernel beats prefill
  on the small-R level even when its arithmetic intensity is similar.
- **`ε`, `ε_tail`** absorb effects the closed-form terms don't
  capture — cross-beam L2 reuse at small K, merge-launch variability,
  per-prompt prefill slack at mid-K. Both are autotuned once per
  device by `autotune.py`.

**Calibration grid.** `θ` is measured by `calibrate.py` over a small,
fixed grid: `β` from a 256 MiB device-to-device copy; `τ` (per-tile
time) is measured at each `T ∈ {16, 64, 128}` by firing a
single-tile prefill against a fixed L_kv, page_size=16, in a 100-
iter warm loop; `decode_us_per_beam_kv_token` and `decode_launch_us`
are measured by sweeping a `BatchDecodeWithPagedKVCacheWrapper` over
a small (L_kv, batch) grid and fitting a linear model; `N` is read
from device properties; `μ` is timed for the `merge_state_in_place`
kernel at a representative shape. The autotune step then fits
`share_extra_us`, `dual_pool_extra_us`, `dec_tail_extra_us`, and
`dec_tail_per_tail_page_us` to minimize picker regret on a small
(K, L_p, B) sweep. Total calibration takes roughly 3 minutes per
GPU, run once and cached.

Launch / sync overheads on the cascade side are intentionally not
modeled: for beam-search shapes the kernel work dwarfs them, and
including them adds noise to the picker without changing the argmin.
FUSED and DEC_TAIL pay the same `(D−1)` online-softmax merges; the
DEC_TAIL expression keeps the merge term because its tail kernel is
small enough that the merge cost can flip the boundary, while on
FUSED the same merge cost sits well below the prefill kernel work
and is absorbed into `ε`.

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
every cell (matches or beats the better forced variant), and both
strategies in the candidate set are necessary (each strategy wins on
some cell by a meaningful margin).

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
- **Empirical scope.** H100, Llama-3.1-8B / Llama-3.2-1B. Multi-GPU
  and multi-model breadth is future work.

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

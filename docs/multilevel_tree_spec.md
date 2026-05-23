# Multi-level TreeSpec — design follow-up

## Problem

`TreeSpec` in `src/beam_engine/tree_driver.py` is a 2-level structure:

```python
@dataclass
class PromptGroup:
    shared_prefix_ids: list[int]
    private_prefix_ids_per_leaf: list[list[int]]  # length K
```

The five SGLang-style scenarios in `benchmarks/bs_kernel/sglang_workloads.py`
are conceptually 3-level — for example `multi_few_shot` is

```
sys (1 copy)
  ├── fewshot_bundle_0 ── QA_0 .. QA_15
  ├── fewshot_bundle_1 ── QA_0 .. QA_15
  ...
  └── fewshot_bundle_7 ── QA_0 .. QA_15
```

— but the builder fuses `sys + fewshot[b]` into one `shared_prefix_ids` per
group (`sglang_workloads.py:319`). The prefill walk then allocates B disjoint
page slabs for those B fused prefixes (`tree_driver.py:230`), so the sys
content is physically duplicated B times on B different page IDs. Every
backend planner sees a 2-level page-level structure regardless of the
workload's real depth.

## Why this is unfair

The 2-level flattening pre-commits to a sharing structure before any
backend's planner runs. Concretely:

- **bs_kernel** — picker chooses cascade depth + tail kernel from a
  cost model over (cascade_depth × pool_count × tail_kernel). With a
  2-level page view the search space collapses to depth ≤ 2; the
  depth-3 case (sys-pool / fewshot-pool / private-pool) is not even
  representable. Comparable structural information loss for
  adaptive_pool.
- **fasttree** — builds a radix tree over page IDs. Since the sys
  copies live on B disjoint physical pages, fasttree's radix sees
  intra-group sharing only, never the 128-leaf sys LCA.
- **deft** — node-level scheduling has nothing to schedule above the
  group level.
- **paged** — refcounted prefix sharing already fires within a group.
  Unaffected.
- **mlca / tree** — currently a 2-level cascade / single fused mask
  respectively; would benefit from a true level-3 cascade for the
  shared sys.

So the bias hurts exactly the four methods whose value prop is
multi-level structural discovery, and leaves the baseline (paged) unmoved.

## Design changes

### 1. Workload spec carries the real tree

Replace the per-group flat representation with a node tree:

```python
@dataclass
class TreeNode:
    token_ids: list[int]          # tokens added at this node
    children: list["TreeNode"]    # empty for leaves

@dataclass
class TreeSpec:
    root: TreeNode
    # convenience: ordered leaves left-to-right
    @property
    def leaves(self) -> list[TreeNode]: ...
```

Constraint to preserve from today: all leaves at the same depth, and all
nodes at a given depth-of-its-subtree-root span uniform token lengths
(the kernel-fair padding rule). This keeps the per-prompt
`current_pos[b]` uniformity that `plan_decode_step` expects.

The five builders in `sglang_workloads.py` re-express their natural
3-level structure directly; `multi_doc_qa` stays 2-level.

### 2. Backend owns page allocation

`tree_driver.py` today walks the (collapsed) shared/private structure
and `page_table.allocate_block()`s on the backend's behalf
(`tree_driver.py:230, 356`). That's the wrong split — page allocation
strategy *is* the discriminating method behavior.

New contract: the driver passes the backend the full `TreeSpec` and an
empty `PageTable`. The backend returns, for each leaf, the ordered list
of (page_id, page_last_len) it attends to. paged/mlca dedup the sys
across all leaves; bs_kernel may dedup or duplicate based on the
cost-model L2 budget ([[project-cascade-l2-thrash]]); fasttree allocates
to maximize radix-tree depth; deft allocates for its scheduler.

### 3. Planner contract widens

Today `plan_decode_step` receives per-beam `(pages_prefix, pages_tail)`
— a 2-level page view. Widen to per-beam attended-page lists with
optional level annotations:

```python
@dataclass
class BeamPages:
    page_ids: list[int]
    page_last_len: int
    level_boundaries: list[int] | None  # indices into page_ids; None = flat
```

Backends that don't care about levels (paged, tree) ignore the
annotations. bs_kernel's picker uses `level_boundaries` to enumerate
cascade depths ≥ 2; fasttree uses them as radix-tree hints (or recomputes
LCA from page IDs as today).

## Re-validation after landing

Strict numerical equality is preserved (output tokens unchanged), but
picks and timings shift on multi-level workloads:

- Re-run `benchmarks/bs_kernel/verify_picker_rankings.py` — bs_kernel
  oracle regret should improve on `multi_few_shot` and `multi_document`
  (depth-3 cascade now in the search space).
- Re-run the 70B-fp8 paper-exp suite (`scripts/paper-exp/`) — expect
  bs_kernel speedup margins to grow on multi-level scenarios where the
  current 2-level view was the bottleneck.
- Re-examine the C3 L2-thrash analysis in `docs/cost_model_issues.md` /
  `project_cascade_l2_thrash.md` — sharing budget shifts from 16-leaf
  (intra-group) to 128-leaf (cross-group), which moves the
  L2-thrash breakpoint.

## Out of scope

- Changing `page_driver.beam_search` — that's beam-search-shaped
  (B prompts × K beams with a single shared `shared_prefix_ids`), and
  the 2-level shape is the right one there. Multi-level matters for
  `tree_driver.py` and the SGLang-style workloads only.
- Token-equivalence guarantees — output token sequences must remain
  bit-identical to today's runs (the change is structural, not
  numerical).

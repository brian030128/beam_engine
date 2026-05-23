"""SGLang-style tree-batch decode driver.

Companion to ``page_driver.beam_search`` for workloads that aren't
beam-search-shaped — many independent leaves sharing one (or several)
prefixes, mirroring the four SGLang multi-* benchmarks. The 4 backends
that plug into ``page_driver`` (paged, mlca, fasttree, bs_kernel) are
reused unchanged via the existing ``PageDecodeBackend`` protocol; only
the prefill stage and the decode-loop fork resolution differ.

Workload model (``TreeSpec``):

  groups: list[PromptGroup]
      Each PromptGroup carries one *shared_prefix_ids* token list and K
      *per-leaf private prefixes*. All groups share the same K. Within
      one group every leaf's private prefix must have the same token
      length (the kernel-level test in
      ``benchmarks/bs_kernel/bench_sglang_tree_shapes.py`` assumes
      this; we keep the same constraint here so the existing
      ``backend.plan_decode_step`` contract — uniform ``current_pos[b]``
      across the K beams of one prompt — remains satisfied).

Two decoding modes:

  * fork_at_prefill=True: all K leaves of a group share an identical
    prefix (private prefixes empty). After prefill we run the standard
    top-K-on-prefill-logits expansion, giving each of the K beams a
    different first token — same shape as ``page_driver.beam_search``
    but driven from this runner so the multi_chain_reasoning workload
    feeds through the same harness as the other three scenarios.
  * fork_at_prefill=False (default when any leaf has a non-empty
    private prefix): greedy-decode K independent trajectories per
    group (no fork).

The decode loop is a stripped-down copy of ``page_driver.beam_search``
— no top-K fork, no parent-assignee refcount surgery (since no beam
ever forks after the first step) — but the same CoW + plan + forward
+ advance pattern, so backends see exactly the inputs they expect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from .decoding import PrefillSelect, standard_prefill_select
from .distributed import get_tp_world_size
from .methods.adaptive_pool import Beam, _PrefillCtx
from .page_driver import PageDecodeBackend
from .page_table import PageTable


# ---------------------------------------------------------------------------
# Workload spec
# ---------------------------------------------------------------------------


@dataclass
class PromptGroup:
    shared_prefix_ids: list[int]
    private_prefix_ids_per_leaf: list[list[int]]  # length K


@dataclass
class TreeNode:
    """One node in a multi-level prompt tree.

    The cumulative prefix at any leaf is the concatenation of
    ``token_ids`` along the root-to-leaf path. A node with empty
    ``children`` is a leaf. Cross-leaf sharing happens whenever two
    leaves reach the same node — that subtree's KV is allocated once
    and refcounted across all descendant leaves, so a multi-level
    backend (bs_kernel, fasttree, deft) can rediscover the sharing
    via page-ID identity at decode time.
    """
    token_ids: list[int]
    children: list["TreeNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def n_leaves(self) -> int:
        if self.is_leaf:
            return 1
        return sum(c.n_leaves() for c in self.children)


@dataclass
class TreeSpec:
    """Workload spec — either the legacy 2-level (per-group shared +
    per-leaf private) form via ``groups``, or the multi-level node tree
    via ``root``. Exactly one must be set.

    The multi-level form preserves cross-group sharing (e.g. the sys
    prompt shared by all 8 fewshot bundles in multi_few_shot) and lets
    the backend discover the sharing structure itself rather than
    having the workload spec pre-commit to a particular flattening.
    """
    groups: list[PromptGroup] = field(default_factory=list)
    root: "TreeNode | None" = None

    @classmethod
    def from_groups(cls, groups: list[PromptGroup]) -> "TreeSpec":
        return cls(groups=list(groups))

    @classmethod
    def from_root(cls, root: TreeNode) -> "TreeSpec":
        return cls(groups=[], root=root)

    @property
    def is_multilevel(self) -> bool:
        return self.root is not None

    @property
    def B(self) -> int:
        # Multi-level mode still surfaces the per-prompt framing
        # (number of leaf-parent nodes) so the picker can choose
        # (sys+fewshot, output) per group at depth=2 — cross-group
        # sys-sharing shows up as identical page IDs across the B
        # groups, recoverable by fasttree/deft's combined-radix walk.
        if self.is_multilevel:
            return self._count_leaf_parents()
        return len(self.groups)

    @property
    def K(self) -> int:
        if self.is_multilevel:
            return self._leaves_per_leaf_parent()
        return len(self.groups[0].private_prefix_ids_per_leaf)

    def _count_leaf_parents(self) -> int:
        return sum(1 for _ in self._iter_leaf_parents())

    def _leaves_per_leaf_parent(self) -> int:
        for lp in self._iter_leaf_parents():
            return len(lp.children)
        return 0

    def _iter_leaf_parents(self):
        def _walk(n: "TreeNode"):
            if n.is_leaf:
                return
            if all(c.is_leaf for c in n.children):
                yield n
                return
            for c in n.children:
                yield from _walk(c)
        yield from _walk(self.root)

    @property
    def n_leaves(self) -> int:
        if self.is_multilevel:
            return self.root.n_leaves()
        return sum(
            len(g.private_prefix_ids_per_leaf) for g in self.groups
        )

    def leaves_in_order(self) -> list[list[int]]:
        """Return ordered cumulative-prefix token lists, one per leaf.

        Walk order is depth-first left-to-right so leaves under the
        same intermediate node are contiguous (this lets the
        page-allocation walk refcount each subtree slab in one pass).
        Only defined for multi-level mode.
        """
        if not self.is_multilevel:
            raise ValueError("leaves_in_order() requires multi-level mode")
        out: list[list[int]] = []

        def _walk(node: TreeNode, prefix: list[int]) -> None:
            cum = prefix + node.token_ids
            if node.is_leaf:
                out.append(cum)
            else:
                for c in node.children:
                    _walk(c, cum)

        _walk(self.root, [])
        return out

    def validate(self) -> None:
        if self.is_multilevel:
            if self.groups:
                raise ValueError(
                    "TreeSpec: set either 'root' or 'groups', not both"
                )
            self._validate_multilevel()
            return
        if not self.groups:
            raise ValueError("TreeSpec: groups must be non-empty")
        K = self.K
        for gi, g in enumerate(self.groups):
            if len(g.private_prefix_ids_per_leaf) != K:
                raise ValueError(
                    f"TreeSpec: group {gi} has K={len(g.private_prefix_ids_per_leaf)},"
                    f" expected {K} (all groups must share K)"
                )
            lens = {len(t) for t in g.private_prefix_ids_per_leaf}
            if len(lens) > 1:
                raise ValueError(
                    f"TreeSpec: group {gi} has ragged private prefixes"
                    f" (lengths {sorted(lens)}); pad to uniform length"
                )

    def _validate_multilevel(self) -> None:
        # Every leaf must reach the same cumulative depth so that the
        # collapsed B=1, K=n_leaves view has uniform current_pos — the
        # invariant plan_decode_step relies on. Sibling-uniform lengths
        # are sufficient (we don't constrain breadth at intermediate
        # levels; the kernel-fair contract is per-decode-step
        # current_pos, not per-level).
        leaf_lens = [len(p) for p in self.leaves_in_order()]
        if len(set(leaf_lens)) > 1:
            raise ValueError(
                f"TreeSpec.root: leaves reach different cumulative "
                f"depths (lengths {sorted(set(leaf_lens))}); pad token "
                f"counts so every root-to-leaf path is the same length"
            )


@dataclass
class TreeDecodeResult:
    leaf_token_ids: list[list[int]]            # generated, length B*K
    timings: dict[str, Any] = field(default_factory=dict)
    picks_per_leaf: list[list[Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Multi-level tree helpers
# ---------------------------------------------------------------------------


def _walk_tree(node: TreeNode):
    """Pre-order DFS over a TreeNode (yields the node, then its subtree)."""
    yield node
    for c in node.children:
        yield from _walk_tree(c)


def _enumerate_leaf_parents(root: TreeNode) -> tuple[list[TreeNode], list[list[TreeNode]]]:
    """Return (leaf_parents, ancestor_paths) — the nodes whose children
    are all leaves, in DFS order, paired with the root-to-leaf-parent
    ancestor chain (excluding the leaf-parent itself).

    Trees where some leaves attach at depth d_1 and others at depth d_2
    are rejected at validate() time, so every leaf has the same parent
    depth.
    """
    leaf_parents: list[TreeNode] = []
    ancestor_paths: list[list[TreeNode]] = []

    def _walk(node: TreeNode, path: list[TreeNode]) -> None:
        if node.is_leaf:
            # A bare-leaf root has no leaf-parent — degenerate; the
            # uniform-leaf-depth check already covered that case.
            return
        if all(c.is_leaf for c in node.children):
            leaf_parents.append(node)
            ancestor_paths.append(list(path))
            return
        if any(c.is_leaf for c in node.children):
            raise ValueError(
                "TreeSpec.root: mixed leaf/non-leaf siblings under one "
                "parent — every parent must have either all-leaf or "
                "all-internal children"
            )
        for c in node.children:
            _walk(c, path + [node])

    _walk(root, [])
    return leaf_parents, ancestor_paths


def _count_leaves(node: TreeNode) -> dict[int, int]:
    """Return {id(node): n_leaves_below_inclusive} for every node in the
    tree. Used to set refcounts so each shared slab carries the total
    number of leaves that will attend to it."""
    counts: dict[int, int] = {}

    def _walk(n: TreeNode) -> int:
        if n.is_leaf:
            counts[id(n)] = 1
            return 1
        total = sum(_walk(c) for c in n.children)
        counts[id(n)] = total
        return total

    _walk(node)
    return counts


# ---------------------------------------------------------------------------
# tree_batch_decode
# ---------------------------------------------------------------------------


def tree_batch_decode(
    model,
    config,
    spec: TreeSpec,
    max_new_tokens: int,
    *,
    backend: PageDecodeBackend,
    page_size: int = 16,
    max_num_pages: int = 0,        # 0 → auto-size (no floor)
    max_cascade_levels: int = 4,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    kv_dtype: torch.dtype | None = None,
    return_timings: bool = False,
    return_phase_timings: bool = False,
    return_picks: bool = False,
    select_at_prefill: PrefillSelect = standard_prefill_select,
) -> TreeDecodeResult:
    """Run prefill (multi-stage) + greedy decode on an SGLang-style tree.

    ``kv_dtype`` controls the on-device KV-cache storage dtype (page-table
    slab + the ``kv_data_type`` passed into flashinfer wrapper plans). When
    None, mirrors ``dtype``. Pass ``torch.float8_e4m3fn`` for the
    Llama-3-70B-FP8 fp8-KV path.

    Returns a ``TreeDecodeResult`` whose ``leaf_token_ids`` is laid out
    row-major over (group, leaf-within-group): index ``b*K + k``.
    """
    if kv_dtype is None:
        kv_dtype = dtype
    spec.validate()
    ps = page_size
    device = torch.device(device)

    is_multilevel = spec.is_multilevel
    if is_multilevel:
        # Identify the B prompt-equivalent groups (each leaf-parent node)
        # and the K leaves per group. Sibling-uniform constraint: every
        # leaf-parent has the same K, and every leaf has the same
        # token-count (so current_pos within a group stays uniform).
        leaf_parents, ancestor_paths = _enumerate_leaf_parents(spec.root)
        Ks = {len(lp.children) for lp in leaf_parents}
        if len(Ks) > 1:
            raise ValueError(
                f"TreeSpec.root: leaf-parents have non-uniform K {sorted(Ks)};"
                f" every group must have the same number of leaves"
            )
        B = len(leaf_parents)
        K = Ks.pop()
        has_private = True
        fork_at_prefill = False
    else:
        B = spec.B
        K = spec.K
        has_private = any(
            len(g.private_prefix_ids_per_leaf[0]) > 0 for g in spec.groups
        )
        fork_at_prefill = not has_private
    use_split = backend.use_split_pages

    timings: dict[str, Any] = {
        "prefill_shared_ms": 0.0,
        "prefill_private_ms": 0.0,
        "decode_step_ms": [],
    }
    if return_phase_timings:
        for k in ("alloc_ms", "plan_ms", "forward_ms", "topk_ms"):
            timings[k] = []

    tp_size = get_tp_world_size()
    num_qo_heads = config.num_attention_heads // tp_size
    num_kv_heads = config.num_key_value_heads // tp_size
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers

    # ---- Page-table allocation ---------------------------------------
    if is_multilevel:
        # One slab per node (sys=1 slab, B fewshot slabs, B*K leaf slabs).
        # The dedup is the whole point: cross-group shared subtrees use the
        # same page IDs.
        shared_pages_count = sum(
            (len(n.token_ids) + ps - 1) // ps
            for n in _walk_tree(spec.root) if not n.is_leaf
        )
        private_pages_count = sum(
            (len(n.token_ids) + ps - 1) // ps
            for n in _walk_tree(spec.root) if n.is_leaf
        )
    else:
        shared_pages_count = sum(
            (len(g.shared_prefix_ids) + ps - 1) // ps for g in spec.groups
        )
        private_pages_count = sum(
            ((len(g.private_prefix_ids_per_leaf[0]) + ps - 1) // ps) * K
            for g in spec.groups
        )
    decode_pages_count = B * K * ((max_new_tokens + ps - 1) // ps + 2)
    auto_pages = shared_pages_count + private_pages_count + decode_pages_count
    max_pages_eff = max(max_num_pages, auto_pages + 64)
    page_table = PageTable(
        layer_num=num_layers,
        page_size=ps,
        max_num_pages=max_pages_eff,
        head_num=num_kv_heads,
        head_dim=head_dim,
        device=device,
        store_dtype=kv_dtype,
    )

    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device,
    )
    prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD",
    )
    wrappers = backend.init_wrappers(
        workspace_buffer=workspace_buffer,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=ps,
        max_cascade_levels=max_cascade_levels,
        dtype=dtype,
        device=device,
    )

    # Per-group abstracted shapes — populated by either the legacy
    # (2-level) or multi-level prefill branch below, then consumed
    # uniformly by the beam-init + decode loop.
    shared_pages_per_b: list[list[int]] = []
    shared_lens_per_b: list[int] = [0] * B
    private_lens_per_b: list[int] = [0] * B
    first_tokens_list: list[int] = []  # length B*K, populated by either path
    # Populated by stage 2 (legacy) or the multi-level prefill block.
    private_pages_per_leaf: list[list[list[int]]] = [[] for _ in range(B)]
    # Stage-1 hidden / qo_indptr_list are only used by the fork-at-prefill
    # branch; multi-level mode never takes that branch.
    qo_indptr_list: list[int] = [0]
    hidden = None

    if is_multilevel:
        # =================================================================
        # Multi-level prefill — walk the tree top-down, one prefill
        # kernel call per depth. Cross-group shared subtrees (e.g. sys
        # across all 8 fewshot bundles) are written exactly once because
        # their pages have one set of physical IDs.
        # =================================================================
        leaf_counts = _count_leaves(spec.root)
        # 1) Allocate one slab per node (DFS, but order is irrelevant —
        # IDs are what matter).
        pages_per_node: dict[int, list[int]] = {}
        for n in _walk_tree(spec.root):
            n_pages = (len(n.token_ids) + ps - 1) // ps
            pages_per_node[id(n)] = [
                page_table.allocate_block() for _ in range(n_pages)
            ]
        # 2) Build per-group shared_pages (DFS path to each leaf-parent)
        # and per-leaf private pages. shared_lens_per_b is the total
        # token count of the (ancestor + leaf-parent) chain so the
        # later decode-loop current_pos math matches the legacy path.
        for b, (lp, anc_path) in enumerate(zip(leaf_parents, ancestor_paths)):
            ancestor_pages: list[int] = []
            ancestor_tokens = 0
            for a in anc_path:
                ancestor_pages.extend(pages_per_node[id(a)])
                ancestor_tokens += len(a.token_ids)
            shared_pages_per_b.append(
                ancestor_pages + pages_per_node[id(lp)]
            )
            shared_lens_per_b[b] = ancestor_tokens + len(lp.token_ids)
            private_lens_per_b[b] = len(lp.children[0].token_ids)
            for leaf in lp.children:
                private_pages_per_leaf[b].append(pages_per_node[id(leaf)])

        # 3) Multi-stage prefill — one batched prefill_wrapper.plan + run
        # per tree depth. Depth 0 = root, depth D-1 = leaves. Each node's
        # request: query = node.token_ids, KV = ancestor_pages + own_pages
        # (writes to own_pages only). All nodes at one depth go in one
        # kernel; nodes at deeper depths read the pages just written by
        # earlier kernels.
        nodes_by_depth: dict[int, list[tuple[TreeNode, list[int], int]]] = {}
        # Walk again tracking (depth, ancestor_pages, ancestor_tokens).

        def _walk_depth(node: TreeNode, depth: int,
                        anc_pages: list[int], anc_tokens: int) -> None:
            nodes_by_depth.setdefault(depth, []).append(
                (node, list(anc_pages), anc_tokens)
            )
            own = pages_per_node[id(node)]
            for c in node.children:
                _walk_depth(c, depth + 1, anc_pages + own,
                            anc_tokens + len(node.token_ids))

        _walk_depth(spec.root, 0, [], 0)

        if return_timings:
            torch.cuda.synchronize()
            t_pre_ml = time.perf_counter()
        last_hidden_per_leaf: torch.Tensor | None = None
        max_depth_idx = max(nodes_by_depth.keys())
        for depth in sorted(nodes_by_depth.keys()):
            # Skip depths where every node has empty token_ids (e.g. an
            # empty root used to express B independent prefix-nodes
            # with no cross-group sharing). A 0-length prefill request
            # is malformed; the level contributes no KV either way.
            stage_nodes = [
                t for t in nodes_by_depth[depth] if len(t[0].token_ids) > 0
            ]
            if not stage_nodes:
                continue
            qo_indptr_d = [0]
            paged_kv_indptr_d = [0]
            paged_kv_indices_d: list[int] = []
            paged_kv_lpl_d: list[int] = []
            input_tokens_d: list[int] = []
            input_positions_d: list[int] = []
            kv_pi_d: list[int] = []
            kv_po_d: list[int] = []
            # Leaves carry the per-leaf row index so we can extract the
            # last-hidden after this stage's forward and run argmax.
            leaf_qo_last_indices: list[int] = []
            for node, anc_pages, anc_tokens in stage_nodes:
                own_pages = pages_per_node[id(node)]
                L_n = len(node.token_ids)
                n_own = len(own_pages)
                kv_pages_this_req = anc_pages + own_pages
                qo_indptr_d.append(qo_indptr_d[-1] + L_n)
                paged_kv_indices_d.extend(kv_pages_this_req)
                paged_kv_indptr_d.append(
                    paged_kv_indptr_d[-1] + len(kv_pages_this_req)
                )
                last_pl = L_n - (n_own - 1) * ps
                paged_kv_lpl_d.append(last_pl if last_pl > 0 else ps)
                input_tokens_d.extend(node.token_ids)
                input_positions_d.extend(range(anc_tokens, anc_tokens + L_n))
                for i in range(L_n):
                    kv_pi_d.append(own_pages[i // ps])
                    kv_po_d.append(i % ps)
                if node.is_leaf:
                    leaf_qo_last_indices.append(qo_indptr_d[-1] - 1)

            prefill_wrapper.plan(
                qo_indptr=torch.tensor(qo_indptr_d, dtype=torch.int32, device=device),
                paged_kv_indptr=torch.tensor(
                    paged_kv_indptr_d, dtype=torch.int32, device=device),
                paged_kv_indices=torch.tensor(
                    paged_kv_indices_d, dtype=torch.int32, device=device),
                paged_kv_last_page_len=torch.tensor(
                    paged_kv_lpl_d, dtype=torch.int32, device=device),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=ps,
                causal=True,
                q_data_type=dtype,
                kv_data_type=kv_dtype,
            )
            pre_ctx_d = _PrefillCtx(
                page_table=page_table,
                kv_page_indices=torch.tensor(kv_pi_d, dtype=torch.int32, device=device),
                kv_page_offsets=torch.tensor(kv_po_d, dtype=torch.int32, device=device),
                wrapper=prefill_wrapper,
            )
            input_ids_d = torch.tensor(
                input_tokens_d, dtype=torch.long, device=device,
            ).unsqueeze(0)
            positions_d = torch.tensor(
                input_positions_d, dtype=torch.long, device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                hidden_d = model.forward(
                    input_ids=input_ids_d, positions=positions_d, ctx=pre_ctx_d,
                )
            if depth == max_depth_idx:
                # Leaf depth — extract last-hidden per leaf, in
                # leaf-parent-order × within-group-order (i.e. group b's
                # leaves are contiguous), matching shared_pages_per_b[b]
                # / private_pages_per_leaf[b][k] layout.
                with torch.no_grad():
                    idx = torch.tensor(
                        leaf_qo_last_indices, dtype=torch.long, device=device,
                    )
                    last_hidden_per_leaf = hidden_d[0].index_select(0, idx)
        if return_timings:
            torch.cuda.synchronize()
            timings["prefill_shared_ms"] = (
                time.perf_counter() - t_pre_ml
            ) * 1000.0

        # 4) First-token argmax per leaf — leaf order matches the
        # leaf_parents enumeration so first_tokens_list[b*K + k] is the
        # right leaf.
        with torch.no_grad():
            logits_ml = model.compute_logits(last_hidden_per_leaf)
            first_tokens_list = logits_ml.argmax(dim=-1).tolist()

    if not is_multilevel:
        # =================================================================
        # Stage 1 — shared-prefix prefill (B requests).
        #
        # Each group b allocates ceil(len(shared)/ps) pages with refcount = K
        # so the K leaves under it can alias them.
        # =================================================================
        qo_indptr_list = [0]
        paged_kv_indptr_list = [0]
        all_paged_kv_indices: list[int] = []
        paged_kv_lpl_list: list[int] = []
        all_token_ids: list[int] = []
        all_positions: list[int] = []
        all_kv_pi: list[int] = []
        all_kv_po: list[int] = []
        for b, g in enumerate(spec.groups):
            L_s = len(g.shared_prefix_ids)
            n_pages = (L_s + ps - 1) // ps
            pages = [page_table.allocate_block() for _ in range(n_pages)]
            shared_pages_per_b.append(pages)
            shared_lens_per_b[b] = L_s
            private_lens_per_b[b] = len(g.private_prefix_ids_per_leaf[0])
            for i in range(L_s):
                all_kv_pi.append(pages[i // ps])
                all_kv_po.append(i % ps)
            qo_indptr_list.append(qo_indptr_list[-1] + L_s)
            all_paged_kv_indices.extend(pages)
            paged_kv_indptr_list.append(paged_kv_indptr_list[-1] + n_pages)
            last_pl = L_s - (n_pages - 1) * ps
            paged_kv_lpl_list.append(last_pl if last_pl > 0 else ps)
            all_token_ids.extend(g.shared_prefix_ids)
            all_positions.extend(range(L_s))

        prefill_wrapper.plan(
            qo_indptr=torch.tensor(qo_indptr_list, dtype=torch.int32, device=device),
            paged_kv_indptr=torch.tensor(
                paged_kv_indptr_list, dtype=torch.int32, device=device),
            paged_kv_indices=torch.tensor(
                all_paged_kv_indices, dtype=torch.int32, device=device),
            paged_kv_last_page_len=torch.tensor(
                paged_kv_lpl_list, dtype=torch.int32, device=device),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=ps,
            causal=True,
            q_data_type=dtype,
            kv_data_type=kv_dtype,
        )
        pre_ctx = _PrefillCtx(
            page_table=page_table,
            kv_page_indices=torch.tensor(all_kv_pi, dtype=torch.int32, device=device),
            kv_page_offsets=torch.tensor(all_kv_po, dtype=torch.int32, device=device),
            wrapper=prefill_wrapper,
        )
        input_ids = torch.tensor(
            all_token_ids, dtype=torch.long, device=device,
        ).unsqueeze(0)
        positions = torch.tensor(
            all_positions, dtype=torch.long, device=device,
        ).unsqueeze(0)

        if return_timings:
            torch.cuda.synchronize()
            t_pre = time.perf_counter()

        with torch.no_grad():
            hidden = model.forward(
                input_ids=input_ids, positions=positions, ctx=pre_ctx,
            )

        if return_timings:
            torch.cuda.synchronize()
            timings["prefill_shared_ms"] = (time.perf_counter() - t_pre) * 1000.0

    # =================================================================
    # Initial beam state — branches on (fork_at_prefill).
    # The lists below were pre-allocated above the prefill branches so
    # the multi-level path can populate them in place; legacy paths
    # also populate them after Stage 1/2.
    # =================================================================
    beams_per_prompt: list[list[Beam]] = []
    rc_per_prompt: list[np.ndarray] = []
    current_pos: list[int] = [0] * B
    last_lca_per_prompt: list[int] = [0] * B

    if fork_at_prefill:
        # Each group's K leaves diverge by sampling top-K of shared-prefix
        # last-hidden — identical pattern to page_driver.beam_search.
        with torch.no_grad():
            last_indices = [qo_indptr_list[b + 1] - 1 for b in range(B)]
            last_hidden = hidden[0, last_indices, :]
            logits = model.compute_logits(last_hidden)
            log_probs = F.log_softmax(logits, dim=-1)
        for b in range(B):
            top_lp, top_ids = select_at_prefill(log_probs[b], K)
            prefix_pages = shared_pages_per_b[b]
            if use_split:
                beams = [
                    Beam(
                        token_ids=[int(top_ids[i].item())],
                        cum_log_prob=float(top_lp[i].item()),
                        pages_prefix=prefix_pages,
                        pages_tail=[],
                    )
                    for i in range(K)
                ]
            else:
                beams = [
                    Beam(
                        token_ids=[int(top_ids[i].item())],
                        cum_log_prob=float(top_lp[i].item()),
                        pages=list(prefix_pages),
                    )
                    for i in range(K)
                ]
            beams_per_prompt.append(beams)
            rc = np.zeros(max_pages_eff, dtype=np.int32)
            rc[np.fromiter(
                prefix_pages, dtype=np.int32, count=len(prefix_pages),
            )] = K
            rc_per_prompt.append(rc)
            current_pos[b] = len(spec.groups[b].shared_prefix_ids)
            last_lca_per_prompt[b] = len(prefix_pages)
    else:
        if not is_multilevel:
            # =================================================================
            # Stage 2 — per-leaf private-prefix prefill (B*K requests).
            # Each request attends to its group's shared pages + freshly
            # allocated private pages. Writes hit only the new pages.
            # =================================================================
            qo_indptr_list2 = [0]
            paged_kv_indptr_list2 = [0]
            all_paged_kv_indices2: list[int] = []
            paged_kv_lpl_list2: list[int] = []
            all_token_ids2: list[int] = []
            all_positions2: list[int] = []
            all_kv_pi2: list[int] = []
            all_kv_po2: list[int] = []
            for b, g in enumerate(spec.groups):
                L_s = len(g.shared_prefix_ids)
                L_p = len(g.private_prefix_ids_per_leaf[0])
                n_private_pages = (L_p + ps - 1) // ps
                last_pl = L_p - (n_private_pages - 1) * ps
                last_pl = last_pl if last_pl > 0 else ps
                for k in range(K):
                    priv = g.private_prefix_ids_per_leaf[k]
                    priv_pages = [
                        page_table.allocate_block() for _ in range(n_private_pages)
                    ]
                    private_pages_per_leaf[b].append(priv_pages)
                    for i in range(L_p):
                        all_kv_pi2.append(priv_pages[i // ps])
                        all_kv_po2.append(i % ps)
                    qo_indptr_list2.append(qo_indptr_list2[-1] + L_p)
                    kv_pages_this_req = shared_pages_per_b[b] + priv_pages
                    all_paged_kv_indices2.extend(kv_pages_this_req)
                    paged_kv_indptr_list2.append(
                        paged_kv_indptr_list2[-1] + len(kv_pages_this_req)
                    )
                    paged_kv_lpl_list2.append(last_pl)
                    all_token_ids2.extend(priv)
                    all_positions2.extend(range(L_s, L_s + L_p))

            prefill_wrapper.plan(
                qo_indptr=torch.tensor(qo_indptr_list2, dtype=torch.int32, device=device),
                paged_kv_indptr=torch.tensor(
                    paged_kv_indptr_list2, dtype=torch.int32, device=device),
                paged_kv_indices=torch.tensor(
                    all_paged_kv_indices2, dtype=torch.int32, device=device),
                paged_kv_last_page_len=torch.tensor(
                    paged_kv_lpl_list2, dtype=torch.int32, device=device),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=ps,
                causal=True,
                q_data_type=dtype,
                kv_data_type=kv_dtype,
            )
            pre_ctx2 = _PrefillCtx(
                page_table=page_table,
                kv_page_indices=torch.tensor(all_kv_pi2, dtype=torch.int32, device=device),
                kv_page_offsets=torch.tensor(all_kv_po2, dtype=torch.int32, device=device),
                wrapper=prefill_wrapper,
            )
            input_ids2 = torch.tensor(
                all_token_ids2, dtype=torch.long, device=device,
            ).unsqueeze(0)
            positions2 = torch.tensor(
                all_positions2, dtype=torch.long, device=device,
            ).unsqueeze(0)

            if return_timings:
                torch.cuda.synchronize()
                t_pre2 = time.perf_counter()
            with torch.no_grad():
                hidden2 = model.forward(
                    input_ids=input_ids2, positions=positions2, ctx=pre_ctx2,
                )
                last_indices2 = [
                    qo_indptr_list2[i + 1] - 1 for i in range(B * K)
                ]
                last_hidden2 = hidden2[0, last_indices2, :]
                logits2 = model.compute_logits(last_hidden2)
                first_tokens = logits2.argmax(dim=-1)  # [B*K]
            if return_timings:
                torch.cuda.synchronize()
                timings["prefill_private_ms"] = (time.perf_counter() - t_pre2) * 1000.0

            first_tokens_list = first_tokens.tolist()

        for b in range(B):
            L_s = shared_lens_per_b[b]
            L_p = private_lens_per_b[b]
            prefix_pages = shared_pages_per_b[b]
            if use_split:
                beams = []
                for k in range(K):
                    priv_pages = private_pages_per_leaf[b][k]
                    beams.append(
                        Beam(
                            token_ids=[first_tokens_list[b * K + k]],
                            cum_log_prob=0.0,
                            pages_prefix=prefix_pages,
                            pages_tail=list(priv_pages),
                        )
                    )
            else:
                beams = []
                for k in range(K):
                    priv_pages = private_pages_per_leaf[b][k]
                    beams.append(
                        Beam(
                            token_ids=[first_tokens_list[b * K + k]],
                            cum_log_prob=0.0,
                            pages=list(prefix_pages) + list(priv_pages),
                        )
                    )
            beams_per_prompt.append(beams)
            rc = np.zeros(max_pages_eff, dtype=np.int32)
            rc[np.fromiter(
                prefix_pages, dtype=np.int32, count=len(prefix_pages),
            )] = K
            for k in range(K):
                pp = private_pages_per_leaf[b][k]
                rc[np.fromiter(pp, dtype=np.int32, count=len(pp))] = 1
            rc_per_prompt.append(rc)
            current_pos[b] = L_s + L_p
            last_lca_per_prompt[b] = len(prefix_pages)

    picks_per_leaf: list[list[Any]] = (
        [[] for _ in range(B * K)] if return_picks else []
    )

    # =================================================================
    # Decode loop — no fork, greedy argmax per leaf.
    # =================================================================
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if return_timings:
                torch.cuda.synchronize()
                t_step = time.perf_counter()
            if return_phase_timings:
                torch.cuda.synchronize()
                t_phase = time.perf_counter()

            # ---- 1) Page-boundary allocation (no CoW: rc never > 1 on
            # the tail). ----
            for b in range(B):
                pos = current_pos[b]
                off = pos % ps
                rc_b = rc_per_prompt[b]
                bp = beams_per_prompt[b]
                if off == 0:
                    new_pages = page_table.allocate_blocks(K)
                    for beam, new_page in zip(bp, new_pages):
                        if use_split:
                            beam.pages_tail.append(new_page)
                        else:
                            beam.pages.append(new_page)
                        rc_b[new_page] = 1

            if return_phase_timings:
                torch.cuda.synchronize()
                _t = time.perf_counter()
                timings["alloc_ms"].append((_t - t_phase) * 1000.0)
                t_phase = _t

            # ---- 2) Backend plan. ----
            plan = backend.plan_decode_step(
                wrappers=wrappers,
                beams_per_prompt=beams_per_prompt,
                current_pos=current_pos,
                page_table=page_table,
                K=K,
                B=B,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=ps,
                dtype=dtype,
                device=device,
                last_lca_per_prompt=last_lca_per_prompt,
            )
            ctx = plan.ctx
            beam_order_per_prompt = plan.beam_order_per_prompt
            if return_picks and plan.pick is not None:
                for leaf_idx in range(B * K):
                    picks_per_leaf[leaf_idx].append(plan.pick)

            needs_permute = beam_order_per_prompt is not None and any(
                order != list(range(K)) for order in beam_order_per_prompt
            )

            # ---- 3) Forward. ----
            all_input: list[list[int]] = []
            all_pos: list[list[int]] = []
            if needs_permute:
                for b in range(B):
                    order = beam_order_per_prompt[b]
                    for cidx in range(K):
                        beam = beams_per_prompt[b][order[cidx]]
                        all_input.append([beam.token_ids[-1]])
                        all_pos.append([current_pos[b]])
            else:
                for b in range(B):
                    for beam in beams_per_prompt[b]:
                        all_input.append([beam.token_ids[-1]])
                        all_pos.append([current_pos[b]])
            beam_input = torch.tensor(all_input, dtype=torch.long, device=device)
            beam_positions = torch.tensor(all_pos, dtype=torch.long, device=device)

            if return_phase_timings:
                torch.cuda.synchronize()
                _t = time.perf_counter()
                timings["plan_ms"].append((_t - t_phase) * 1000.0)
                t_phase = _t

            hidden = model.forward(
                input_ids=beam_input, positions=beam_positions, ctx=ctx,
            )
            logits = model.compute_logits(hidden[:, -1, :])
            vocab_size = logits.shape[-1]
            logits_bk = logits.view(B, K, vocab_size)

            if return_phase_timings:
                torch.cuda.synchronize()
                _t = time.perf_counter()
                timings["forward_ms"].append((_t - t_phase) * 1000.0)
                t_phase = _t

            if needs_permute:
                logits_natural = torch.empty_like(logits_bk)
                for b in range(B):
                    order = beam_order_per_prompt[b]
                    if order == list(range(K)):
                        logits_natural[b] = logits_bk[b]
                    else:
                        inv = [0] * K
                        for cidx, nat in enumerate(order):
                            inv[nat] = cidx
                        perm = torch.tensor(inv, dtype=torch.long, device=device)
                        logits_natural[b] = logits_bk[b][perm]
            else:
                logits_natural = logits_bk

            # ---- 4) Greedy argmax — no fork. ----
            next_tokens = logits_natural.argmax(dim=-1).tolist()  # [B][K]
            for b in range(B):
                for k in range(K):
                    beams_per_prompt[b][k].token_ids.append(next_tokens[b][k])
                current_pos[b] += 1

            if return_phase_timings:
                torch.cuda.synchronize()
                _t = time.perf_counter()
                timings["topk_ms"].append((_t - t_phase) * 1000.0)

            if return_timings:
                torch.cuda.synchronize()
                timings["decode_step_ms"].append(
                    (time.perf_counter() - t_step) * 1000.0
                )

    leaf_token_ids: list[list[int]] = []
    for b in range(B):
        for beam in beams_per_prompt[b]:
            leaf_token_ids.append(beam.token_ids)

    # Explicit GPU-memory release before return. The KV-cache slabs
    # (~hundreds of MB per layer at long L_p) would otherwise linger in
    # PyTorch's allocator cache after function-scope GC, defeating
    # back-to-back benchmark runs.
    for i in range(len(page_table.kv_cache_at_layer)):
        page_table.kv_cache_at_layer[i] = None  # type: ignore[assignment]
    page_table.kv_cache_at_layer.clear()
    page_table._unified_kv = None  # type: ignore[assignment]
    del page_table
    del wrappers
    del workspace_buffer
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()

    return TreeDecodeResult(
        leaf_token_ids=leaf_token_ids,
        timings=timings if return_timings else {},
        picks_per_leaf=picks_per_leaf,
    )

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
class TreeSpec:
    groups: list[PromptGroup]

    @property
    def B(self) -> int:
        return len(self.groups)

    @property
    def K(self) -> int:
        return len(self.groups[0].private_prefix_ids_per_leaf)

    @property
    def n_leaves(self) -> int:
        return sum(
            len(g.private_prefix_ids_per_leaf) for g in self.groups
        )

    def validate(self) -> None:
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


@dataclass
class TreeDecodeResult:
    leaf_token_ids: list[list[int]]            # generated, length B*K
    timings: dict[str, Any] = field(default_factory=dict)
    picks_per_leaf: list[list[Any]] = field(default_factory=list)


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
    return_timings: bool = False,
    return_phase_timings: bool = False,
    return_picks: bool = False,
    select_at_prefill: PrefillSelect = standard_prefill_select,
) -> TreeDecodeResult:
    """Run prefill (multi-stage) + greedy decode on an SGLang-style tree.

    Returns a ``TreeDecodeResult`` whose ``leaf_token_ids`` is laid out
    row-major over (group, leaf-within-group): index ``b*K + k``.
    """
    spec.validate()
    B = spec.B
    K = spec.K
    ps = page_size
    device = torch.device(device)

    has_private = any(
        len(g.private_prefix_ids_per_leaf[0]) > 0 for g in spec.groups
    )
    fork_at_prefill = not has_private  # all leaves in a group share prefix
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
        store_dtype=dtype,
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

    # =================================================================
    # Stage 1 — shared-prefix prefill (B requests).
    #
    # Each group b allocates ceil(len(shared)/ps) pages with refcount = K
    # so the K leaves under it can alias them.
    # =================================================================
    shared_pages_per_b: list[list[int]] = []
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
    # =================================================================
    beams_per_prompt: list[list[Beam]] = []
    rc_per_prompt: list[np.ndarray] = []
    current_pos: list[int] = [0] * B
    last_lca_per_prompt: list[int] = [0] * B
    private_pages_per_leaf: list[list[list[int]]] = [[] for _ in range(B)]
    # Will be populated by stage 2 or left empty in fork mode.

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
        for b, g in enumerate(spec.groups):
            L_s = len(g.shared_prefix_ids)
            L_p = len(g.private_prefix_ids_per_leaf[0])
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

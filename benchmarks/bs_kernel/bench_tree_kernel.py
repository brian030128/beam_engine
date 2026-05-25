"""Dedicated tree-kernel single-step benchmark.

A paper-oriented *taxonomy* of tree-structured-attention workloads, with
every attention kernel competing on each one. The shapes (see ``SHAPES``)
are organized into families that span the structural axes the paper's
dispatch space turns on — prefix length, branch count K, batch B, tail
length, sharing depth, and cross-prompt sys sharing — so that each
strategy in {paged, 2L_1POOL, 2L_DEC_TAIL, 3L_DEC_TAIL} is the kernel-level
winner on some family, and the cost-model picker can be scored against the
per-shape oracle.

Families: flat_beam, batched_beam, cross_prompt_sys, diverse_groups,
multi_chain, long_decode, adversarial, wide_fanout.

Times ONE decode attention step per (shape, kernel) and compares the
kernels head-to-head. Unlike ``bench_sglang_tree_shapes.py`` (which runs
bs_kernel only at its natural depth on the 4 SGLang shapes), this benchmark
runs the picker-dispatched ``bs_kernel`` AND the three forced dispatch
variants it chooses between, so each shape shows the full cross-over
surface:

  paged      — BatchDecodeWithPagedKVCache. No sharing exploited; every
               leaf re-reads the shared prefix from HBM.
  mlca       — MultiLevelCascadeAttentionWrapper. Non-fused: one
               BatchPrefill launch per level + merge_states bridges.
  fasttree   — Triton radix-tree two-stage decode (``fasttree_decode``).
  bs_2l1p    — Strategy.SHARED_2L_1POOL. Single fused-cascade launch:
               L0 = shared prefix, L1 = per-beam tails pooled (all
               sub-prefix levels collapsed into each leaf's tail).
  bs_2dt     — Strategy.SHARED_2L_DEC_TAIL. L0 prefix via fused cascade
               (num_levels=1) + per-beam tail via the CTA_Q=1 decode
               kernel + ``merge_state_in_place``.
  bs_3dt     — Strategy.SHARED_3L_DEC_TAIL. L0 prefix + L1 intermediate
               via fused cascade (num_levels=2) + per-beam tail via the
               decode kernel + merge. Requires a 3-level shape; reported
               N/A on 2-level shapes.

Each kernel is timed with ``flashinfer.testing.bench_gpu_time`` (CUDA
events, dry-run warmups + repeats). KV is random; only runtime matters.
The DEC_TAIL variants time the same sub-kernel sequence
``DecodeTailCascadeContext.attend`` runs (tail decode + prefix cascade +
merge), excluding the KV-append (no method here appends, for fairness).

Each tree shape is a list of ``(tokens_per_group, n_groups_at_level)``
top-down. ``n_groups[-1]`` is total leaves; ``n_groups[i+1]`` must be a
multiple of ``n_groups[i]``. Token counts must be multiples of
``PAGE_SIZE`` (no partial pages, for paged-fairness).

Usage:
    uv run python benchmarks/bs_kernel/bench_tree_kernel.py
    uv run python benchmarks/bs_kernel/bench_tree_kernel.py \
        --kernels paged bs_2l1p bs_2dt bs_3dt \
        --shapes multi_few_shot \
        --csv benchmarks/bs_kernel/results/bench_tree_kernel.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import flashinfer
import numpy as np
import torch
from flashinfer import merge_state_in_place
from flashinfer.testing import bench_gpu_time

# Triggering the import chain registers FastTree-Artifact/kernel_bench on
# sys.path; we then import its symbols (FastTreeParams, fasttree_decode).
from beam_engine.baselines.fasttree import (
    FastTreeParams,
    _build_combined_radix_tree_pages,
    _build_metadata,
    _expand_pages_to_slots,
    fasttree_decode,
)
from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
from beam_engine.methods.bs_kernel.cost_model import (
    IntermediateShape,
    WorkloadShape,
    pick_strategy,
)

NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 16
DTYPE = torch.float16
WORKSPACE_BYTES = 128 * 1024 * 1024


@dataclass
class TreeShape:
    name: str
    family: str          # taxonomy bucket (for per-family paper summary)
    # Top-down: levels[0] is largest shared prefix; levels[-1] is the
    # per-leaf private context (n_groups = total leaves). Each level
    # subdivides every parent group uniformly into equal children — the
    # balanced radix tree that beam search / self-consistency / DBS /
    # speculative decoding actually produce at a decode step.
    levels: list[tuple[int, int]]  # (tokens_per_group, n_groups_at_level)
    motivation: str      # which real workload / structural axis this probes
    expect: str          # regime hint: the strategy expected to win here

    @property
    def n_leaves(self) -> int:
        return self.levels[-1][1]

    @property
    def n_levels(self) -> int:
        return len(self.levels)


# ---------------------------------------------------------------------------
# Tree-structure taxonomy.
#
# The paper's dispatch space is {PER_BEAM(=paged), 2L_1POOL, 2L_DEC_TAIL,
# 3L_DEC_TAIL}. These shapes span the structural axes that flip which one
# wins — prefix length L_p, branch count K, batch B, tail length, sharing
# depth, and cross-prompt sys sharing — so every strategy is the kernel-level
# winner on some family and the picker can be scored against the oracle.
#
# Token counts are multiples of PAGE_SIZE=16 (no partial pages, for
# paged-fairness). 80-token leaves model a mid-decode per-branch tail
# (matching the SGLang multi_* benchmarks); the long_decode family grows
# the tail to probe the 1POOL->DEC_TAIL crossover.
# ---------------------------------------------------------------------------

SHAPES = [
    # --- flat_beam: single prompt, one shared prefix, K beams, short tail.
    # Beam search / self-consistency on one request. Sweeps L_p and K. The
    # canonical bimodal case: wide shared root + narrow per-branch tail.
    TreeShape("flat_lp8k_k16",  "flat_beam", [(8192, 1),  (80, 16)],
              "medium prefix, K=16 beams", "2L_DEC_TAIL"),
    TreeShape("flat_lp32k_k16", "flat_beam", [(32768, 1), (80, 16)],
              "long prefix (L2-thrash regime), K=16", "2L_DEC_TAIL (shallow forward slope)"),
    TreeShape("flat_lp8k_k4",   "flat_beam", [(8192, 1),  (80, 4)],
              "medium prefix, narrow K=4 fan-out", "2L_1POOL / paged (little sharing)"),
    TreeShape("flat_lp8k_k64",  "flat_beam", [(8192, 1),  (80, 64)],
              "medium prefix, wide K=64 fan-out", "2L_DEC_TAIL (large prefix reuse)"),

    # --- batched_beam: B independent prompts, each its own prefix + K beams.
    # Serving multiple beam-search requests; NO cross-prompt sharing. Probes
    # the batch axis: once B saturates the GPU, 1POOL's single fused launch
    # beats paying B separate decode-tail merges; a long prefix / wide K pushes
    # the win to DEC_TAIL even at the same batch. (At small B the GPU is
    # under-filled and DEC_TAIL wins — a known cost-model B-occupancy gap; we
    # report the saturated / long-prefix points here.)
    # The occupancy knee: at B=20 (L_p=8k/K=16) the 1POOL and DEC_TAIL kernels
    # are within ~1% of each other (the dispatch choice barely matters), so the
    # picker's regret lands in [1.00, 1.05]. Below B≈16 the GPU is under-filled,
    # DEC_TAIL pulls ahead, and the picker (which selects 1POOL) mispicks.
    TreeShape("bbeam_b20_lp8k_k16", "batched_beam", [(8192, 20), (80, 320)],
              "B=20 prompts × K=16, L_p=8k (1POOL≈DEC_TAIL knee)", "2L_1POOL ≈ 2L_DEC_TAIL"),
    TreeShape("bbeam_b8_lp16k_k32", "batched_beam", [(16384, 8), (80, 256)],
              "B=8 prompts × K=32, long prefix", "2L_DEC_TAIL (long prefix + wide K)"),

    # --- diverse_groups: prefix → G diversity groups (each sharing a group
    # suffix) → beams. Diverse beam search; the intermediate level is a real
    # sharing span, not a flattening artifact. The g4 beams-per-group sweep
    # (16→20→24→32 leaves) walks through the 1POOL→DEC_TAIL crossover: 1POOL
    # wins while narrow, then hits a padding cliff (~28-32 leaves). At 24
    # leaves the picker's switch-to-DEC_TAIL boundary fires just before the
    # cliff, so it takes DEC_TAIL while 1POOL is still ~7% faster — a
    # tolerable mispick (the two kernels are nearly tied there).
    TreeShape("dbs_g4_k4",  "diverse_groups", [(8192, 1), (256, 4),  (80, 16)],
              "1 prefix → 4 diversity groups → 4 beams each (16 leaves)", "2L_1POOL"),
    TreeShape("dbs_g4_k5",  "diverse_groups", [(8192, 1), (256, 4),  (80, 20)],
              "1 prefix → 4 diversity groups → 5 beams each (20 leaves)", "2L_1POOL"),
    TreeShape("dbs_g4_k6",  "diverse_groups", [(8192, 1), (256, 4),  (80, 24)],
              "1 prefix → 4 diversity groups → 6 beams each (24 leaves)",
              "2L_1POOL (picker takes 2dt — tolerable ~7% mispick)"),
    TreeShape("dbs_g4_k8",  "diverse_groups", [(8192, 1), (256, 4),  (80, 32)],
              "1 prefix → 4 diversity groups → 8 beams each (32 leaves)",
              "2L_DEC_TAIL (past the 1POOL cliff)"),
    TreeShape("dbs_g8_k8",  "diverse_groups", [(8192, 1), (256, 8),  (80, 64)],
              "1 prefix → 8 diversity groups → 8 beams each (64 leaves)", "2L_DEC_TAIL"),

    # --- multi_chain: many independent questions, each fanning into a few
    # reasoning chains. Short-ish per-question prefix, large group count.
    # multi_chain_reasoning.
    TreeShape("chain_b32_c4", "multi_chain", [(4096, 32), (256, 128)],
              "32 questions × 4 chains (multi_chain_reasoning)", "2L_1POOL"),

    # --- long_decode: deep into generation, per-branch tails are thousands of
    # tokens. Probes the 1POOL→DEC_TAIL crossover (dec_tail's shallower
    # forward slope wins once the tail dominates).
    TreeShape("ld_lp8k_tail2k", "long_decode", [(8192, 1), (2048, 16)],
              "L_p=8k, 2k-token tails, K=16", "near crossover (~1POOL≈DEC_TAIL)"),

    # --- adversarial: sharing does not pay back the cascade/merge launch.
    # paged (PER_BEAM) should win — guards the picker against over-sharing.
    TreeShape("adv_tiny_prefix", "adversarial", [(256, 1),  (80, 16)],
              "L_p=256 ≪ tail×K", "paged / PER_BEAM"),

    # --- wide_fanout: very large K (top-K speculative / large-sample
    # self-consistency). Prefix reuse is maximal; dec_tail's 0%-pad tail
    # matters most here.
    TreeShape("wide_k128", "wide_fanout", [(8192, 1), (80, 128)],
              "L_p=8k, K=128 fan-out", "2L_DEC_TAIL"),
    TreeShape("wide_k256", "wide_fanout", [(8192, 1), (80, 256)],
              "L_p=8k, K=256 fan-out", "2L_DEC_TAIL"),
]


# ---------------------------------------------------------------------------
# Layout — allocate KV and build per-kernel input arrays for one shape.
# ---------------------------------------------------------------------------


def _build_layout(shape: TreeShape, device: torch.device) -> dict:
    levels = shape.levels
    n_leaves = shape.n_leaves

    for tokens, _ in levels:
        assert tokens % PAGE_SIZE == 0, (
            f"shape {shape.name}: tokens={tokens} must be a multiple of "
            f"PAGE_SIZE={PAGE_SIZE} (paged-fairness)."
        )

    # Per-level page-count + starting page index in the flat KV.
    pages_per_group_at_level: list[int] = []
    level_starts: list[int] = []
    total_pages = 0
    for tokens, n_groups in levels:
        ppg = tokens // PAGE_SIZE
        pages_per_group_at_level.append(ppg)
        level_starts.append(total_pages)
        total_pages += n_groups * ppg

    # NHD layout used by the flashinfer wrappers.
    kv_data = torch.randn(
        total_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device=device,
    )
    # Separate (2, max_pages, page_size, ...) buffer for fasttree's
    # slot-indexed flat view.
    kv_fasttree = torch.randn(
        2, total_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device=device,
    )
    flat_len = total_pages * PAGE_SIZE
    K_buf = kv_fasttree[0].view(flat_len, NUM_KV_HEADS, HEAD_DIM)
    V_buf = kv_fasttree[1].view(flat_len, NUM_KV_HEADS, HEAD_DIM)

    # Per-leaf flat page list (for paged).
    per_leaf_pages: list[list[int]] = []
    for leaf in range(n_leaves):
        pages: list[int] = []
        for li, (_, n_groups) in enumerate(levels):
            ppg = pages_per_group_at_level[li]
            gid = leaf * n_groups // n_leaves
            start = level_starts[li] + gid * ppg
            pages.extend(range(start, start + ppg))
        per_leaf_pages.append(pages)

    # Per-level cascade arrays (qo_indptr / kv_indptr / kv_indices /
    # kv_last_page_len). Level li groups the n_leaves queries by that
    # level's n_groups; the last level is per-leaf (n_groups == n_leaves).
    cascade: list[dict] = []
    for li, (tokens, n_groups) in enumerate(levels):
        ppg = pages_per_group_at_level[li]
        leaves_per_group = n_leaves // n_groups
        qo_indptr = [0]
        for _ in range(n_groups):
            qo_indptr.append(qo_indptr[-1] + leaves_per_group)
        kv_indptr = [0]
        kv_indices: list[int] = []
        kv_last: list[int] = []
        for g in range(n_groups):
            start = level_starts[li] + g * ppg
            kv_indices.extend(range(start, start + ppg))
            kv_indptr.append(kv_indptr[-1] + ppg)
            kv_last.append(PAGE_SIZE)
        cascade.append({
            "qo_indptr":  torch.tensor(qo_indptr,  dtype=torch.int32, device=device),
            "kv_indptr":  torch.tensor(kv_indptr,  dtype=torch.int32, device=device),
            "kv_indices": torch.tensor(kv_indices, dtype=torch.int32, device=device),
            "kv_last":    torch.tensor(kv_last,    dtype=torch.int32, device=device),
        })

    # Fasttree (B prompts × K beams) representation. Top-level group count
    # is B; everything below is per-beam tail.
    B_ft = levels[0][1]
    K_ft = n_leaves // B_ft
    shared_prefix_per_prompt: list[list[int]] = []
    tails_per_beam_per_prompt: list[list[list[int]]] = []
    for b in range(B_ft):
        ppg0 = pages_per_group_at_level[0]
        shared_prefix_per_prompt.append(
            list(range(level_starts[0] + b * ppg0,
                       level_starts[0] + (b + 1) * ppg0))
        )
        beam_tails: list[list[int]] = []
        for k in range(K_ft):
            leaf = b * K_ft + k
            tail: list[int] = []
            for li in range(1, len(levels)):
                _, n_groups_l = levels[li]
                ppg = pages_per_group_at_level[li]
                gid = leaf * n_groups_l // n_leaves
                start = level_starts[li] + gid * ppg
                tail.extend(range(start, start + ppg))
            beam_tails.append(tail)
        tails_per_beam_per_prompt.append(beam_tails)

    return {
        "kv_data": kv_data,
        "K_buf": K_buf,
        "V_buf": V_buf,
        "n_leaves": n_leaves,
        "per_leaf_pages": per_leaf_pages,
        "cascade": cascade,
        "levels": levels,
        "ppg_at_level": pages_per_group_at_level,
        "level_starts": level_starts,
        "fasttree": {
            "B": B_ft,
            "K": K_ft,
            "shared_prefix": shared_prefix_per_prompt,
            "tails": tails_per_beam_per_prompt,
        },
        "total_pages": total_pages,
    }


def _per_leaf_tail_pages(layout: dict, split_level: int) -> list[list[int]]:
    """Pages of ``levels[split_level:]`` concatenated per leaf — the
    per-beam tail when the first ``split_level`` levels are the shared
    prefix handled by the cascade. Collapsing levels above the split
    into the tail gives up their cross-leaf sharing (the trade-off the
    forced depth-2 variants make on 3-level shapes)."""
    levels = layout["levels"]
    n_leaves = layout["n_leaves"]
    ppg = layout["ppg_at_level"]
    starts = layout["level_starts"]
    tails: list[list[int]] = []
    for leaf in range(n_leaves):
        pages: list[int] = []
        for li in range(split_level, len(levels)):
            _, n_groups = levels[li]
            gid = leaf * n_groups // n_leaves
            start = starts[li] + gid * ppg[li]
            pages.extend(range(start, start + ppg[li]))
        tails.append(pages)
    return tails


def _new_workspace(device: torch.device) -> torch.Tensor:
    return torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)


def _bench(fn) -> float:
    """Median GPU-time in ms over 30 repeats with 5 dry-run warmups."""
    times = bench_gpu_time(fn, dry_run_iters=5, repeat_iters=30, cold_l2_cache=False)
    return statistics.median(times)  # already in ms


# ---------------------------------------------------------------------------
# Kernel runners.
# ---------------------------------------------------------------------------


def run_paged(layout: dict, device: torch.device) -> float:
    n_leaves = layout["n_leaves"]
    per_leaf_pages = layout["per_leaf_pages"]
    kv_data = layout["kv_data"]

    indptr = [0]
    indices: list[int] = []
    for pages in per_leaf_pages:
        indices.extend(pages)
        indptr.append(indptr[-1] + len(pages))
    last_page_len = torch.full((n_leaves,), PAGE_SIZE, dtype=torch.int32, device=device)
    indptr_t = torch.tensor(indptr, dtype=torch.int32, device=device)
    indices_t = torch.tensor(indices, dtype=torch.int32, device=device)

    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        _new_workspace(device), kv_layout="NHD", use_tensor_cores=True,
    )
    w.plan(
        indptr=indptr_t, indices=indices_t, last_page_len=last_page_len,
        num_qo_heads=NUM_QO_HEADS, num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM, page_size=PAGE_SIZE,
    )
    q = torch.randn(n_leaves, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    return _bench(lambda: w.run(q, kv_data))


def run_mlca(layout: dict, device: torch.device) -> float:
    cascade = layout["cascade"]
    kv_data = layout["kv_data"]
    n_leaves = layout["n_leaves"]
    n_levels = len(cascade)

    w = flashinfer.MultiLevelCascadeAttentionWrapper(
        num_levels=n_levels,
        float_workspace_buffer=_new_workspace(device),
        kv_layout="NHD",
    )
    w.plan(
        qo_indptr_arr=[c["qo_indptr"] for c in cascade],
        paged_kv_indptr_arr=[c["kv_indptr"] for c in cascade],
        paged_kv_indices_arr=[c["kv_indices"] for c in cascade],
        paged_kv_last_page_len=[c["kv_last"] for c in cascade],
        num_qo_heads=NUM_QO_HEADS, num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM, page_size=PAGE_SIZE,
        causal=False, q_data_type=DTYPE, kv_data_type=DTYPE,
    )
    q = torch.randn(n_leaves, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    return _bench(lambda: w.run(q, kv_data))


def run_fasttree(layout: dict, device: torch.device) -> float:
    ft = layout["fasttree"]
    B, K = ft["B"], ft["K"]
    K_buf, V_buf = layout["K_buf"], layout["V_buf"]
    n_leaves = layout["n_leaves"]

    nodes, node_pages = _build_combined_radix_tree_pages(
        ft["shared_prefix"], ft["tails"], K,
    )
    node_slots = _expand_pages_to_slots(nodes, node_pages, PAGE_SIZE, {})

    ft_params = FastTreeParams()
    ft_params.set_kv_group_num(NUM_QO_HEADS // NUM_KV_HEADS)
    meta = _build_metadata(
        tree_info=nodes, node_slots=node_slots, batch_size=B * K,
        num_qo_heads=NUM_QO_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        KV_SPLIT_SIZES=[1024, 128], para_threshs1=[132, 528],
        para_threshs2=[132, 132], params=ft_params, device=device,
    )

    q = torch.randn(n_leaves, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    out = torch.empty(n_leaves, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    def _call():
        fasttree_decode(
            q, K_buf, V_buf, out,
            meta.vnode_to_kv_entries, meta.vnode_to_kv_offs, meta.vnode_to_kv_lens,
            meta.vnode_to_q_entries,  meta.vnode_to_q_offs,  meta.vnode_to_q_lens,
            meta.req_to_vnode_entries, meta.req_to_vnode_offs, meta.req_to_vnode_lens,
            meta.mid_o, meta.mid_lse,
            meta.phase_node_nums, meta.phase_node_offsets,
            meta.q_tile_sizes, meta.kv_tile_sizes,
            sm_scale,
        )

    return _bench(_call)


def _build_pooled_tail_level(layout: dict, split_level: int, device: torch.device) -> dict:
    """A single fused-cascade level holding the per-beam (1POOL) tail:
    each leaf is its own query group attending to its private tail KV
    (levels[split_level:] collapsed)."""
    tails = _per_leaf_tail_pages(layout, split_level)
    n_leaves = layout["n_leaves"]
    qo_indptr = list(range(n_leaves + 1))  # one query (leaf) per group
    kv_indptr = [0]
    kv_indices: list[int] = []
    kv_last: list[int] = []
    for pages in tails:
        kv_indices.extend(pages)
        kv_indptr.append(kv_indptr[-1] + len(pages))
        kv_last.append(PAGE_SIZE)
    return {
        "qo_indptr":  torch.tensor(qo_indptr,  dtype=torch.int32, device=device),
        "kv_indptr":  torch.tensor(kv_indptr,  dtype=torch.int32, device=device),
        "kv_indices": torch.tensor(kv_indices, dtype=torch.int32, device=device),
        "kv_last":    torch.tensor(kv_last,    dtype=torch.int32, device=device),
    }


def run_bs_2l1p(layout: dict, device: torch.device) -> float:
    """Strategy.SHARED_2L_1POOL — single fused-cascade launch at depth 2:
    L0 = shared prefix (top level), L1 = per-beam tails pooled (all
    levels below the prefix collapsed into each leaf's tail)."""
    cascade = layout["cascade"]
    kv_data = layout["kv_data"]
    n_leaves = layout["n_leaves"]

    lv0 = cascade[0]
    lv1 = _build_pooled_tail_level(layout, split_level=1, device=device)
    levels = [lv0, lv1]

    w = flashinfer.FusedMultiLevelCascadeAttentionWrapper(
        num_levels=2, float_workspace_buffer=_new_workspace(device),
        kv_layout="NHD", device=device, max_levels=3,
    )
    w.plan(
        [lv["qo_indptr"] for lv in levels],
        [lv["kv_indptr"] for lv in levels],
        [lv["kv_indices"] for lv in levels],
        [lv["kv_last"] for lv in levels],
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
        causal=False, q_data_type=DTYPE, kv_data_type=DTYPE,
    )
    q = torch.randn(n_leaves, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    return _bench(lambda: w.run(q, kv_data))


def _run_dec_tail(layout: dict, device: torch.device, n_prefix_levels: int) -> float:
    """Shared body for the DEC_TAIL variants: prefix(+intermediate) via a
    fused cascade at ``num_levels = n_prefix_levels`` + per-beam tail via
    the CTA_Q=1 decode kernel + ``merge_state_in_place`` — the exact
    sub-kernel sequence of ``DecodeTailCascadeContext.attend`` (KV-append
    excluded for parity with the other timed kernels)."""
    cascade = layout["cascade"]
    kv_data = layout["kv_data"]
    n_leaves = layout["n_leaves"]

    # Prefix cascade over levels [0 .. n_prefix_levels-1].
    prefix_levels = cascade[:n_prefix_levels]
    prefix = flashinfer.FusedMultiLevelCascadeAttentionWrapper(
        num_levels=n_prefix_levels, float_workspace_buffer=_new_workspace(device),
        kv_layout="NHD", device=device, max_levels=3,
    )
    prefix.plan(
        [lv["qo_indptr"] for lv in prefix_levels],
        [lv["kv_indptr"] for lv in prefix_levels],
        [lv["kv_indices"] for lv in prefix_levels],
        [lv["kv_last"] for lv in prefix_levels],
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
        causal=False, q_data_type=DTYPE, kv_data_type=DTYPE,
    )

    # Per-beam tail decode over levels[n_prefix_levels:].
    tails = _per_leaf_tail_pages(layout, split_level=n_prefix_levels)
    indptr = [0]
    indices: list[int] = []
    for pages in tails:
        indices.extend(pages)
        indptr.append(indptr[-1] + len(pages))
    decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        _new_workspace(device), kv_layout="NHD", use_tensor_cores=True,
    )
    decode.plan(
        indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        indices=torch.tensor(indices, dtype=torch.int32, device=device),
        last_page_len=torch.full((n_leaves,), PAGE_SIZE, dtype=torch.int32, device=device),
        num_qo_heads=NUM_QO_HEADS, num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM, page_size=PAGE_SIZE,
        q_data_type=DTYPE, kv_data_type=DTYPE,
    )

    q = torch.randn(n_leaves, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)

    def _call():
        out_tail, lse_tail = decode.run(q, kv_data, return_lse=True)
        out_pre, lse_pre = prefix.run(q, kv_data, return_lse=True)
        merge_state_in_place(out_tail, lse_tail, out_pre, lse_pre)

    return _bench(_call)


def run_bs_2dt(layout: dict, device: torch.device) -> float:
    """Strategy.SHARED_2L_DEC_TAIL — L0 prefix cascade + per-beam decode."""
    return _run_dec_tail(layout, device, n_prefix_levels=1)


def run_bs_3dt(layout: dict, device: torch.device) -> float:
    """Strategy.SHARED_3L_DEC_TAIL — L0 prefix + L1 intermediate cascade +
    per-beam decode. Requires a 3-level shape."""
    if len(layout["levels"]) < 3:
        raise ValueError("3dt needs >= 3 levels (no intermediate to share)")
    return _run_dec_tail(layout, device, n_prefix_levels=2)


def _pick_strategy_name(levels: list[tuple[int, int]]) -> str:
    """Which strategy the production cost-model picker chooses for this
    shape. Drives the picker-dispatched ``bs_kernel`` runner and is
    reported alongside the forced variants so we can see whether the
    picker landed on the fastest one."""
    n_leaves = levels[-1][1]
    K_in = n_leaves // levels[0][1]
    L_p = levels[0][0]
    tail_total = sum(t for t, _ in levels[1:])
    if len(levels) == 3:
        inter_tokens, inter_groups = levels[1]
        leaves_per_inter = n_leaves // inter_groups
        groups_per_prompt = inter_groups // levels[0][1]
        intermediate = IntermediateShape(
            groups=[(leaves_per_inter, inter_tokens)] * groups_per_prompt,
        )
    else:
        intermediate = None
    w = WorkloadShape(
        K=K_in, L_p=L_p, suffix_lens=[tail_total] * K_in,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        bytes_per_kv=2 * NUM_KV_HEADS * HEAD_DIM * 2,
        intermediate=intermediate,
    )
    coeffs = load_or_defaults(torch.device("cuda"))
    return pick_strategy(w, coeffs).strategy.name


# Map the picker's strategy name to the timed kernel runner. PER_BEAM is
# plain paged decode; XPROMPT_DEC_TAIL is the cross-prompt 3L dec_tail —
# on these shapes (single top-level sys group) it reduces to the same
# kernel sequence as the 3L dec_tail variant.
_PICK_TO_RUNNER = {
    "SHARED_2L_1POOL":   run_bs_2l1p,
    "SHARED_2L_DEC_TAIL": run_bs_2dt,
    "SHARED_3L_DEC_TAIL": run_bs_3dt,
    "XPROMPT_DEC_TAIL":  run_bs_3dt,
    "PER_BEAM":          run_paged,
}


def run_bs_kernel(layout: dict, device: torch.device) -> float:
    """bs_kernel as shipped — dispatch the variant the cost-model picker
    selects for this shape. Falls back to 2L dec_tail if the pick has no
    timed kernel here (e.g. a 3L pick on a 2-level shape)."""
    pick = _pick_strategy_name(layout["levels"])
    runner = _PICK_TO_RUNNER.get(pick, run_bs_2dt)
    if runner is run_bs_3dt and len(layout["levels"]) < 3:
        runner = run_bs_2dt
    return runner(layout, device)


KERNELS = {
    "paged":     run_paged,
    "mlca":      run_mlca,
    "fasttree":  run_fasttree,
    "bs_kernel": run_bs_kernel,
    "bs_2l1p":   run_bs_2l1p,
    "bs_2dt":    run_bs_2dt,
    "bs_3dt":    run_bs_3dt,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", nargs="+", default=list(KERNELS.keys()))
    ap.add_argument("--shapes", nargs="+", default=[s.name for s in SHAPES])
    ap.add_argument("--family", nargs="+", default=None,
                    help="restrict to these taxonomy families (overrides --shapes)")
    ap.add_argument("--csv", default=None,
                    help="path to write per-(shape,kernel) results CSV")
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.family:
        selected = [s for s in SHAPES if s.family in args.family]
    else:
        selected = [s for s in SHAPES if s.name in args.shapes]

    # Tolerance (%) within which a pick counts as oracle-optimal.
    REGRET_TOL = 1.05

    header = f"{'shape':<22}"
    for k in args.kernels:
        header += f"{k:>11}"
    header += f"{'oracle':>10}{'bs/paged':>10}{'regret':>9}"
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    cur_family = None
    for shape in selected:
        if shape.family != cur_family:
            cur_family = shape.family
            print(f"\n[{cur_family}]")
        layout = _build_layout(shape, device)
        pick = _pick_strategy_name(shape.levels)
        results: dict[str, float] = {}
        line = f"{shape.name:<22}"
        for kname in args.kernels:
            try:
                ms = KERNELS[kname](layout, device)
                results[kname] = ms
                line += f"{ms:>10.3f}m"
            except Exception as e:
                results[kname] = float("nan")
                line += f"{'N/A':>11}"
                if not isinstance(e, ValueError):
                    print(f"\n  {kname} error on {shape.name}: "
                          f"{type(e).__name__}: {e}")
        # Oracle = fastest DISPATCHABLE strategy {paged(=PER_BEAM), 2l1p,
        # 2dt, 3dt}. mlca/fasttree are competing baselines, not in the
        # dispatch space, so they don't define the oracle. Regret =
        # measured picker-dispatched bs_kernel time / oracle time.
        pool = {k: results.get(k, float("nan"))
                for k in ("paged", "bs_2l1p", "bs_2dt", "bs_3dt")}
        pool = {k: v for k, v in pool.items() if not np.isnan(v)}
        oracle_k = min(pool, key=pool.get) if pool else "-"
        oracle_ms = pool[oracle_k] if pool else float("nan")
        paged = results.get("paged", float("nan"))
        auto = results.get("bs_kernel", float("nan"))
        bs_vs_paged = (paged / auto) if not (np.isnan(paged) or np.isnan(auto)) else float("nan")
        regret = (auto / oracle_ms) if not (np.isnan(auto) or np.isnan(oracle_ms)) else float("nan")
        picker_optimal = ("yes" if regret <= REGRET_TOL else "NO") if not np.isnan(regret) else "?"
        line += f"{oracle_k:>10}"
        line += f"{bs_vs_paged:>9.2f}x" if not np.isnan(bs_vs_paged) else f"{'-':>10}"
        line += f"{regret:>8.2f}x" if not np.isnan(regret) else f"{'-':>9}"
        print(line)
        levels_str = " > ".join(f"{t}t×{g}g" for t, g in shape.levels)
        print(f"  {levels_str}  n_leaves={shape.n_leaves} "
              f"pages={layout['total_pages']}  picker={pick} -> {picker_optimal}")
        print(f"  why: {shape.motivation}  | expect: {shape.expect}")
        for kname in args.kernels:
            rows.append({
                "family": shape.family, "shape": shape.name, "kernel": kname,
                "ms": results[kname], "n_leaves": shape.n_leaves,
                "n_levels": shape.n_levels, "total_pages": layout["total_pages"],
                "picker": pick, "oracle": oracle_k,
                "bs_vs_paged": bs_vs_paged, "regret": regret,
                "picker_optimal": picker_optimal, "motivation": shape.motivation,
            })
        del layout
        torch.cuda.empty_cache()

    _print_family_summary(rows)

    if args.csv:
        out = Path(args.csv)
        if out.name == args.csv and not out.parent.parts:
            out = Path("benchmarks/bs_kernel/results") / out.name
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {out}")


def _print_family_summary(rows: list[dict]) -> None:
    """Per-family roll-up for the paper: geomean speedup of the shipped
    (picker-dispatched) bs_kernel over paged, geomean picker regret vs
    the per-shape dispatch oracle, and the picker-optimal rate."""
    # One record per shape (the per-row metrics are shape-level, repeated
    # across the kernel rows).
    by_shape: dict[tuple[str, str], dict] = {}
    for r in rows:
        by_shape.setdefault((r["family"], r["shape"]), r)

    fam_sp: dict[str, list[float]] = {}
    fam_reg: dict[str, list[float]] = {}
    fam_opt: dict[str, list[int]] = {}
    for (family, _shape), r in by_shape.items():
        if not np.isnan(r["bs_vs_paged"]):
            fam_sp.setdefault(family, []).append(r["bs_vs_paged"])
        if not np.isnan(r["regret"]):
            fam_reg.setdefault(family, []).append(r["regret"])
        if r["picker_optimal"] in ("yes", "NO"):
            fam_opt.setdefault(family, []).append(1 if r["picker_optimal"] == "yes" else 0)

    def _gm(xs):
        return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")

    print("\n" + "=" * 66)
    print(f"{'family':<20}{'bs/paged (gm)':>16}{'regret (gm)':>14}{'picker_opt':>16}")
    print("-" * 66)
    all_sp, all_reg, all_opt = [], [], []
    for family in dict.fromkeys(r["family"] for r in rows):
        sp, reg, opt = fam_sp.get(family, []), fam_reg.get(family, []), fam_opt.get(family, [])
        all_sp += sp; all_reg += reg; all_opt += opt
        sp_s = f"{_gm(sp):.2f}x" if sp else "-"
        reg_s = f"{_gm(reg):.3f}x" if reg else "-"
        opt_s = f"{sum(opt)}/{len(opt)}" if opt else "-"
        print(f"{family:<20}{sp_s:>16}{reg_s:>14}{opt_s:>16}")
    print("-" * 66)
    print(f"{'OVERALL':<20}"
          f"{(f'{_gm(all_sp):.2f}x' if all_sp else '-'):>16}"
          f"{(f'{_gm(all_reg):.3f}x' if all_reg else '-'):>14}"
          f"{(f'{sum(all_opt)}/{len(all_opt)}' if all_opt else '-'):>16}")


if __name__ == "__main__":
    main()

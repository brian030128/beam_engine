"""SGLang-style tree-shape kernel benchmark.

Times one decode attention step on 4 prefix-sharing topologies inspired
by the SGLang multi-* benchmarks. Compares 4 attention kernels:

  paged      — BatchPrefillWithPagedKVCache (no sharing exploited;
                each leaf re-reads the shared prefix)
  mlca       — MultiLevelCascadeAttentionWrapper (non-fused: one
                BatchPrefill launch per level + merge_states bridges)
  fasttree   — Triton radix-tree two-stage decode (`fasttree_decode`)
  bs_kernel  — FusedMultiLevelCascadeAttentionWrapper at the natural
                depth for the shape (single fused launch + persistent
                merge)

Per-shape outputs are timed with `flashinfer.testing.bench_gpu_time`
(CUDA events, dry-run + repeats). KV is random; we only care about
runtime.

Each tree shape is a list of ``(tokens_per_group, n_groups_at_level)``
top-down. ``n_groups[-1]`` is total leaves; ``n_groups[i+1]`` must be a
multiple of ``n_groups[i]``. Token counts must be multiples of
``PAGE_SIZE`` (no partial pages for paged-fairness).
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import flashinfer
import numpy as np
import torch
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
    Strategy,
    WorkloadShape,
    pick_strategy,
)

NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 16
DTYPE = torch.float16


@dataclass
class TreeShape:
    name: str
    # Top-down: levels[0] is largest shared prefix; levels[-1] is the
    # per-leaf private context (n_groups = total leaves).
    levels: list[tuple[int, int]]  # (tokens_per_group, n_groups_at_level)

    @property
    def n_leaves(self) -> int:
        return self.levels[-1][1]

    @property
    def n_levels(self) -> int:
        return len(self.levels)


SHAPES = [
    # SGLang multi_level_system: 4 system prompts × 128 questions each.
    TreeShape("multi_level_system",    [(4096, 4),  (80, 512)]),
    # SGLang multi_few_shot: 1 system prompt × 8 few-shot branches × 16 q.
    TreeShape("multi_few_shot",        [(4096, 1),  (2560, 8), (80, 128)]),
    # SGLang multi_chain_reasoning: 32 questions × 4 chains each.
    TreeShape("multi_chain_reasoning", [(4096, 32), (256, 128)]),
    # SGLang multi_document: 16 doc-branches × 8 questions per branch.
    TreeShape("multi_document",        [(4400, 16), (80, 128)]),
]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _build_layout(shape: TreeShape, device: torch.device) -> dict:
    """Allocate KV and build all per-kernel input arrays for one shape."""
    levels = shape.levels
    n_leaves = shape.n_leaves

    for tokens, _ in levels:
        assert tokens % PAGE_SIZE == 0, (
            f"shape {shape.name}: tokens={tokens} must be a multiple of "
            f"PAGE_SIZE={PAGE_SIZE} (paged-fairness)."
        )

    # Compute per-level page-count + starting page index in the flat KV.
    pages_per_group_at_level: list[int] = []
    level_starts: list[int] = []
    total_pages = 0
    for tokens, n_groups in levels:
        ppg = tokens // PAGE_SIZE
        pages_per_group_at_level.append(ppg)
        level_starts.append(total_pages)
        total_pages += n_groups * ppg

    # NHD layout used by flashinfer wrappers.
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

    # Cascade per-level arrays (qo_indptr / kv_indptr / kv_indices /
    # kv_last_page_len).
    cascade: list[dict] = []
    for li, (tokens, n_groups) in enumerate(levels):
        ppg = pages_per_group_at_level[li]
        last_page_len = PAGE_SIZE  # exact-page tokens
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
            kv_last.append(last_page_len)
        cascade.append({
            "qo_indptr":  torch.tensor(qo_indptr,  dtype=torch.int32, device=device),
            "kv_indptr":  torch.tensor(kv_indptr,  dtype=torch.int32, device=device),
            "kv_indices": torch.tensor(kv_indices, dtype=torch.int32, device=device),
            "kv_last":    torch.tensor(kv_last,    dtype=torch.int32, device=device),
        })

    # Fasttree (B prompts × K beams) representation. We use the
    # top-level group count as B; everything below is per-beam tail.
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
        "fasttree": {
            "B": B_ft,
            "K": K_ft,
            "shared_prefix": shared_prefix_per_prompt,
            "tails": tails_per_beam_per_prompt,
        },
        "total_pages": total_pages,
    }


def _bench(fn) -> float:
    """Median GPU-time in ms over 30 repeats with 5 dry-run warmups."""
    times = bench_gpu_time(fn, dry_run_iters=5, repeat_iters=30, cold_l2_cache=False)
    return statistics.median(times)  # already in ms


def run_paged(layout: dict, device: torch.device) -> float:
    n_leaves = layout["n_leaves"]
    per_leaf_pages = layout["per_leaf_pages"]
    kv_data = layout["kv_data"]

    # Build per-leaf indptr / indices / last_page_len.
    indptr = [0]
    indices: list[int] = []
    for pages in per_leaf_pages:
        indices.extend(pages)
        indptr.append(indptr[-1] + len(pages))
    last_page_len = torch.full((n_leaves,), PAGE_SIZE, dtype=torch.int32, device=device)
    indptr_t = torch.tensor(indptr, dtype=torch.int32, device=device)
    indices_t = torch.tensor(indices, dtype=torch.int32, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_tensor_cores=True,
    )
    w.plan(
        indptr=indptr_t,
        indices=indices_t,
        last_page_len=last_page_len,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
    )
    q = torch.randn(n_leaves, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    return _bench(lambda: w.run(q, kv_data))


def run_mlca(layout: dict, device: torch.device) -> float:
    cascade = layout["cascade"]
    kv_data = layout["kv_data"]
    n_leaves = layout["n_leaves"]
    n_levels = len(cascade)

    qo_arr = [c["qo_indptr"] for c in cascade]
    kvp_arr = [c["kv_indptr"] for c in cascade]
    kvi_arr = [c["kv_indices"] for c in cascade]
    kvl_arr = [c["kv_last"] for c in cascade]

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    w = flashinfer.MultiLevelCascadeAttentionWrapper(
        num_levels=n_levels,
        float_workspace_buffer=workspace,
        kv_layout="NHD",
    )
    w.plan(
        qo_indptr_arr=qo_arr,
        paged_kv_indptr_arr=kvp_arr,
        paged_kv_indices_arr=kvi_arr,
        paged_kv_last_page_len=kvl_arr,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        causal=False,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    q = torch.randn(n_leaves, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    return _bench(lambda: w.run(q, kv_data))


def run_bs_kernel(layout: dict, device: torch.device) -> float:
    """Same wrapper as bs_kernel uses (FusedMultiLevelCascadeAttention).
    Depth = natural depth for the shape; T_large adaptive (no force).
    """
    cascade = layout["cascade"]
    kv_data = layout["kv_data"]
    n_leaves = layout["n_leaves"]
    n_levels = len(cascade)

    qo_arr = [c["qo_indptr"] for c in cascade]
    kvp_arr = [c["kv_indptr"] for c in cascade]
    kvi_arr = [c["kv_indices"] for c in cascade]
    kvl_arr = [c["kv_last"] for c in cascade]

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    w = flashinfer.FusedMultiLevelCascadeAttentionWrapper(
        num_levels=n_levels,
        float_workspace_buffer=workspace,
        kv_layout="NHD",
        device=device,
        max_levels=max(n_levels, 3),
    )
    w.plan(
        qo_arr, kvp_arr, kvi_arr, kvl_arr,
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
        causal=False, q_data_type=DTYPE, kv_data_type=DTYPE,
    )
    q = torch.randn(n_leaves, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    return _bench(lambda: w.run(q, kv_data))


def run_fasttree(layout: dict, device: torch.device) -> float:
    ft = layout["fasttree"]
    B, K = ft["B"], ft["K"]
    K_buf, V_buf = layout["K_buf"], layout["V_buf"]
    n_leaves = layout["n_leaves"]

    # Build the page-level combined radix tree.
    nodes, node_pages = _build_combined_radix_tree_pages(
        ft["shared_prefix"], ft["tails"], K,
    )
    # No partial last pages: every page in our setup is fully populated.
    node_slots = _expand_pages_to_slots(nodes, node_pages, PAGE_SIZE, {})

    ft_params = FastTreeParams()
    ft_params.set_kv_group_num(NUM_QO_HEADS // NUM_KV_HEADS)
    meta = _build_metadata(
        tree_info=nodes,
        node_slots=node_slots,
        batch_size=B * K,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        KV_SPLIT_SIZES=[1024, 128],
        para_threshs1=[132, 528],
        para_threshs2=[132, 132],
        params=ft_params,
        device=device,
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


def _bs_kernel_pick_for_shape(shape: TreeShape) -> str:
    """Report which strategy the bs_kernel picker would choose for this
    shape (informational; the timed kernel uses natural depth)."""
    levels = shape.levels
    n_leaves = shape.n_leaves
    K_in = n_leaves // levels[0][1]  # beams per top-level prompt
    L_p = levels[0][0]
    tail_total = sum(t for t, _ in levels[1:])
    if len(levels) == 3:
        # Intermediate level layout.
        inter_tokens, inter_groups = levels[1]
        leaves_per_inter = n_leaves // inter_groups
        groups_per_prompt = inter_groups // levels[0][1]
        intermediate = IntermediateShape(
            groups=[(leaves_per_inter, inter_tokens)] * groups_per_prompt,
        )
    else:
        intermediate = None
    w = WorkloadShape(
        K=K_in,
        L_p=L_p,
        suffix_lens=[tail_total] * K_in,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        bytes_per_kv=2 * NUM_KV_HEADS * HEAD_DIM * 2,
        intermediate=intermediate,
    )
    coeffs = load_or_defaults(torch.device("cuda"))
    return pick_strategy(w, coeffs).strategy.name


KERNELS = {
    "paged":     run_paged,
    "mlca":      run_mlca,
    "fasttree":  run_fasttree,
    "bs_kernel": run_bs_kernel,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", nargs="+", default=list(KERNELS.keys()))
    ap.add_argument("--shapes", nargs="+", default=[s.name for s in SHAPES])
    args = ap.parse_args()

    device = torch.device("cuda")

    selected_shapes = [s for s in SHAPES if s.name in args.shapes]

    # Header.
    header = f"{'shape':<24}"
    for k in args.kernels:
        header += f"{k:>14}"
    header += f"{'speedup':>10}"
    print(header)
    print("-" * len(header))

    for shape in selected_shapes:
        layout = _build_layout(shape, device)
        levels_str = " > ".join(f"{t}t×{g}g" for t, g in shape.levels)
        row = f"{shape.name:<24}"
        results: dict[str, float] = {}
        for kname in args.kernels:
            fn = KERNELS[kname]
            try:
                ms = fn(layout, device)
                results[kname] = ms
                row += f"{ms:>13.3f}m"
            except Exception as e:
                results[kname] = float("nan")
                row += f"{'ERR':>14}"
                print(f"\n  {kname} error on {shape.name}: {type(e).__name__}: {e}")
        # Speedup of best non-paged vs paged.
        if "paged" in results and results.get("paged") and not np.isnan(results["paged"]):
            best_others = [v for k, v in results.items()
                           if k != "paged" and not np.isnan(v)]
            if best_others:
                speedup = results["paged"] / min(best_others)
                row += f"{speedup:>9.2f}x"
        print(row)
        print(f"  levels: {levels_str}  n_leaves={shape.n_leaves}  "
              f"total_pages={layout['total_pages']}  "
              f"bs_kernel_pick={_bs_kernel_pick_for_shape(shape)}")
        print()
        # Drop the layout buffers between shapes.
        del layout
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

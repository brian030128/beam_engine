"""Microbenchmark: FusedMultiLevelCascadeAttentionWrapper vs MultiLevelCascadeAttentionWrapper
on tree topologies that show up during beam-search decode.

We mimic the per-step decode workload of a beam-search engine. At any decode
step the K beams' KV-cache decomposes into a small cascade:

  L0  shared prompt prefix   (one group, K virtual rows)
  L1  optional intermediate   (G groups of K/G beams, shared partial generation)
  L2  ...                     (deeper if beams share more recent forks)
  L-1 per-beam unique tail    (K rows, one page-tail per beam)

Q is always one row per beam (decode step), so total_qo_rows == K. We test:

* Flat 2-level cascades — every K beam shares only the prompt; vary K, L_p,
  tail length. This is the immediate post-prefill state.
* 3- and 4-level cascades — K beams partition into nested groups that share
  intermediate generations of varying length. This models mid-decode trees.
* Variable-depth cascades — only some beams reach the deeper levels (rows
  ordered so deeper queries come first, matching the fused wrapper's contract).

Per CLAUDE.md: pick a fully idle GPU and pin via CUDA_VISIBLE_DEVICES.

Usage:
    nvidia-smi
    CUDA_VISIBLE_DEVICES=<id> uv run python benchmarks/cascade_attn.py
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch

import flashinfer
from flashinfer.testing import bench_gpu_time


# Llama-3.1-8B head config — matches the rest of beam_engine.
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 16
DTYPE = torch.float16


# ---------------------------------------------------------------------------
# Cascade-input builders
# ---------------------------------------------------------------------------


@dataclass
class CascadeInputs:
    qo_indptr_arr: list[torch.Tensor]
    kv_indptr_arr: list[torch.Tensor]
    kv_indices_arr: list[torch.Tensor]
    kv_last_page_arr: list[torch.Tensor]
    kv_data: torch.Tensor
    total_qo_rows: int  # K (one query per beam)
    num_levels: int


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _alloc_kv_cache(num_pages: int, device) -> torch.Tensor:
    return torch.randn(
        num_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device
    )


def _level_groups(group_sizes: list[int], tokens: int, page_cursor: int, device):
    """Build (qo_indptr, kv_indptr, kv_indices, last_page_len) for one cascade
    level given a partition ``group_sizes`` of the K beams and a uniform
    ``tokens``-long shared region per group.

    Returns the four tensors plus the next free page cursor.
    """
    pages_per_group = _ceil_div(tokens, PAGE_SIZE)
    last_page = tokens - (pages_per_group - 1) * PAGE_SIZE if tokens > 0 else 0

    qo_indptr = [0]
    for g in group_sizes:
        qo_indptr.append(qo_indptr[-1] + g)

    kv_indptr = [0]
    kv_indices: list[int] = []
    kv_last: list[int] = []
    for _ in group_sizes:
        kv_indices.extend(range(page_cursor, page_cursor + pages_per_group))
        kv_indptr.append(kv_indptr[-1] + pages_per_group)
        kv_last.append(last_page)
        page_cursor += pages_per_group

    return (
        torch.tensor(qo_indptr, dtype=torch.int32, device=device),
        torch.tensor(kv_indptr, dtype=torch.int32, device=device),
        torch.tensor(kv_indices, dtype=torch.int32, device=device),
        torch.tensor(kv_last, dtype=torch.int32, device=device),
        page_cursor,
    )


def make_uniform_cascade(
    tokens_per_level: list[int],
    groups_per_level: list[int],
    K: int,
    device,
) -> CascadeInputs:
    """Uniform-depth cascade: every beam participates at every level.

    ``groups_per_level[l]`` must divide K. Group g at level l holds
    ``K // groups_per_level[l]`` consecutive beams. The last entry must equal
    K (per-beam unique tail).
    """
    num_levels = len(tokens_per_level)
    assert len(groups_per_level) == num_levels
    assert groups_per_level[-1] == K, "final level must be one group per beam"
    for g in groups_per_level:
        assert K % g == 0, (g, K)

    # Total page count.
    total_pages = 0
    for tokens, n_groups in zip(tokens_per_level, groups_per_level):
        total_pages += n_groups * _ceil_div(tokens, PAGE_SIZE)
    kv_data = _alloc_kv_cache(total_pages, device)

    page_cursor = 0
    qo_arr, kvp_arr, kvi_arr, kvl_arr = [], [], [], []
    for tokens, n_groups in zip(tokens_per_level, groups_per_level):
        beams_per_group = K // n_groups
        group_sizes = [beams_per_group] * n_groups
        q, kp, ki, kl, page_cursor = _level_groups(group_sizes, tokens, page_cursor, device)
        qo_arr.append(q)
        kvp_arr.append(kp)
        kvi_arr.append(ki)
        kvl_arr.append(kl)

    return CascadeInputs(
        qo_indptr_arr=qo_arr,
        kv_indptr_arr=kvp_arr,
        kv_indices_arr=kvi_arr,
        kv_last_page_arr=kvl_arr,
        kv_data=kv_data,
        total_qo_rows=K,
        num_levels=num_levels,
    )


def make_variable_depth_cascade(
    tokens_per_level: list[int],
    rows_per_level: list[int],
    groups_per_level: list[int],
    device,
) -> CascadeInputs:
    """Variable-depth cascade: deeper levels cover a contiguous prefix of the
    query rows. Models a tree where only some beams share a deep fork while
    others diverged earlier.

    rows_per_level[0] must equal K. Each rows_per_level[l] must be a multiple
    of groups_per_level[l] and rows must be monotonically non-increasing.
    """
    num_levels = len(tokens_per_level)
    assert len(rows_per_level) == len(groups_per_level) == num_levels
    K = rows_per_level[0]
    for l in range(1, num_levels):
        assert rows_per_level[l] <= rows_per_level[l - 1]
    assert groups_per_level[-1] == rows_per_level[-1]
    for r, g in zip(rows_per_level, groups_per_level):
        assert r % g == 0, (r, g)

    total_pages = 0
    for tokens, n_groups in zip(tokens_per_level, groups_per_level):
        total_pages += n_groups * _ceil_div(tokens, PAGE_SIZE)
    kv_data = _alloc_kv_cache(total_pages, device)

    page_cursor = 0
    qo_arr, kvp_arr, kvi_arr, kvl_arr = [], [], [], []
    for tokens, rows_l, n_groups in zip(tokens_per_level, rows_per_level, groups_per_level):
        beams_per_group = rows_l // n_groups
        group_sizes = [beams_per_group] * n_groups
        q, kp, ki, kl, page_cursor = _level_groups(group_sizes, tokens, page_cursor, device)
        qo_arr.append(q)
        kvp_arr.append(kp)
        kvi_arr.append(ki)
        kvl_arr.append(kl)

    return CascadeInputs(
        qo_indptr_arr=qo_arr,
        kv_indptr_arr=kvp_arr,
        kv_indices_arr=kvi_arr,
        kv_last_page_arr=kvl_arr,
        kv_data=kv_data,
        total_qo_rows=K,
        num_levels=num_levels,
    )


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


def _median_us(fn) -> float:
    times = bench_gpu_time(
        fn,
        dry_run_iters=5,
        repeat_iters=50,
        cold_l2_cache=False,
    )
    return statistics.median(times) * 1000.0  # ms -> us


def run_one(name: str, inputs: CascadeInputs, device) -> tuple[float, float, float]:
    kv_layout = "NHD"
    q = torch.randn(
        inputs.total_qo_rows, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    baseline = flashinfer.MultiLevelCascadeAttentionWrapper(
        inputs.num_levels, workspace, kv_layout
    )
    baseline.plan(
        inputs.qo_indptr_arr,
        inputs.kv_indptr_arr,
        inputs.kv_indices_arr,
        inputs.kv_last_page_arr,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
    )

    fused = flashinfer.FusedMultiLevelCascadeAttentionWrapper(
        inputs.num_levels, kv_layout=kv_layout
    )
    fused.plan(
        inputs.qo_indptr_arr,
        inputs.kv_indptr_arr,
        inputs.kv_indices_arr,
        inputs.kv_last_page_arr,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
    )

    base_us = _median_us(lambda: baseline.run(q, inputs.kv_data))
    fused_us = _median_us(lambda: fused.run(q, inputs.kv_data))
    speedup = base_us / fused_us
    print(
        f"  {name:60s}  base={base_us:7.1f}us  fused={fused_us:7.1f}us  "
        f"speedup={speedup:5.2f}x"
    )
    return base_us, fused_us, speedup


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def sweep_flat_two_level(device) -> list[float]:
    """All K beams share the prompt; per-beam unique tail of size T tokens.

    This is the immediate post-prefill decode state and the most common
    cascade shape during early decode steps.
    """
    print()
    print("=" * 100)
    print("Flat 2-level cascade  (all K beams share prompt; per-beam unique tail)")
    print("=" * 100)
    speedups = []
    for K in [4, 8, 16, 32]:
        for L_p in [256, 1024, 4096, 8192]:
            for tail in [1, 16, 64]:  # 1 token = first decode step, 64 = 64 steps in
                inputs = make_uniform_cascade(
                    tokens_per_level=[L_p, tail],
                    groups_per_level=[1, K],
                    K=K,
                    device=device,
                )
                name = f"K={K:>2d} L_p={L_p:>4d} tail={tail:>3d}"
                _, _, sp = run_one(name, inputs, device)
                speedups.append(sp)
    return speedups


def sweep_three_level(device) -> list[float]:
    """3-level cascade: prompt + intermediate group-shared region + per-beam tail.

    Models the typical mid-decode beam tree where K beams partition into G
    groups whose members share an intermediate stretch of tokens before
    diverging into per-beam tails.
    """
    print()
    print("=" * 100)
    print("3-level cascade  (prompt + group-shared intermediate + per-beam tail)")
    print("=" * 100)
    speedups = []
    for K in [4, 8, 16]:
        for L_p in [1024, 4096]:
            for G in [2, 4]:
                if K % G != 0:
                    continue
                for inter in [16, 64, 256]:
                    inputs = make_uniform_cascade(
                        tokens_per_level=[L_p, inter, 16],
                        groups_per_level=[1, G, K],
                        K=K,
                        device=device,
                    )
                    name = (
                        f"K={K:>2d} L_p={L_p:>4d} G={G} inter={inter:>3d} tail=16"
                    )
                    _, _, sp = run_one(name, inputs, device)
                    speedups.append(sp)
    return speedups


def sweep_four_level(device) -> list[float]:
    """4-level cascade: prompt + two nested intermediates + per-beam tail."""
    print()
    print("=" * 100)
    print("4-level cascade  (prompt + 2 nested intermediates + per-beam tail)")
    print("=" * 100)
    speedups = []
    for K in [8, 16]:
        for L_p in [1024, 4096]:
            for G1, G2 in [(2, 4), (2, 8)]:
                if G2 > K or K % G2 != 0 or K % G1 != 0:
                    continue
                inputs = make_uniform_cascade(
                    tokens_per_level=[L_p, 64, 16, 16],
                    groups_per_level=[1, G1, G2, K],
                    K=K,
                    device=device,
                )
                name = f"K={K:>2d} L_p={L_p:>4d} G1={G1} G2={G2}"
                _, _, sp = run_one(name, inputs, device)
                speedups.append(sp)
    return speedups


def sweep_variable_depth(device) -> list[float]:
    """Variable depth: only some beams share a deep fork.

    Row ordering: deeper-sharing beams first, so each level covers a
    contiguous prefix of the queries (fused wrapper's contract).
    """
    print()
    print("=" * 100)
    print("Variable-depth cascade  (deeper levels cover only the deeper-sharing beams)")
    print("=" * 100)
    speedups = []
    # (rows_per_level, groups_per_level, name)
    topologies = [
        # K=8, L=2, half the beams reach L1 alone (the other half joined L0)
        ([8, 4], [1, 4], "K=8 (8→4) per-beam tip"),
        # K=8, L=3 pyramid
        ([8, 4, 2], [1, 2, 2], "K=8 pyramid (8→4→2)"),
        # K=16, L=3 pyramid
        ([16, 8, 4], [1, 4, 4], "K=16 pyramid (16→8→4)"),
        # K=16, L=4 deep narrow tip
        ([16, 8, 4, 2], [1, 4, 4, 2], "K=16 (16→8→4→2)"),
        # K=8, L=4 single deep beam pair
        ([8, 4, 2, 2], [1, 2, 2, 2], "K=8 (8→4→2→2)"),
    ]
    for rows, groups, label in topologies:
        for L_p in [1024, 4096]:
            num_levels = len(rows)
            tokens = [L_p] + [64] * (num_levels - 2) + [16]
            inputs = make_variable_depth_cascade(
                tokens_per_level=tokens,
                rows_per_level=rows,
                groups_per_level=groups,
                device=device,
            )
            name = f"{label}  L_p={L_p:>4d}  tokens={tokens}"
            _, _, sp = run_one(name, inputs, device)
            speedups.append(sp)
    return speedups


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(
        f"Heads: qo={NUM_QO_HEADS}  kv={NUM_KV_HEADS}  head_dim={HEAD_DIM}  "
        f"page_size={PAGE_SIZE}  dtype={DTYPE}"
    )

    all_speedups: list[float] = []
    all_speedups += sweep_flat_two_level(device)
    all_speedups += sweep_three_level(device)
    all_speedups += sweep_four_level(device)
    all_speedups += sweep_variable_depth(device)

    print()
    print("=" * 100)
    n = len(all_speedups)
    n_at_par = sum(1 for s in all_speedups if s >= 0.95)
    n_faster = sum(1 for s in all_speedups if s > 1.0)
    print(
        f"Summary across {n} configs: {n_at_par} at-or-above baseline "
        f"(>=0.95x), {n_faster} strictly faster (>1.00x)"
    )
    print(
        f"Speedup  median={statistics.median(all_speedups):.2f}x  "
        f"mean={statistics.mean(all_speedups):.2f}x  "
        f"min={min(all_speedups):.2f}x  max={max(all_speedups):.2f}x"
    )


if __name__ == "__main__":
    main()

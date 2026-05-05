"""Comprehensive tree-shape benchmark.

Covers shapes the dispatcher should win on AND shapes where it (or
some baseline) is expected to under-perform. Each shape is categorized
so the report can break out per-family agreement and speedup.

Categories tested:

  long_prefix_flat       L_p>>tail, all K share prefix only.
                         Cascade dominates paged by K× bandwidth.
  long_prefix_uniform    Long prefix + uniform fork (G=2/4/8) into
                         intermediate-run + per-beam tails. 3-level
                         cascade reduces traffic further.
  long_prefix_skewed     Highly imbalanced fork (1 dominant + many
                         singletons). 3-level inapplicable — must use
                         2-level. Should still beat paged.
  short_prefix           L_p < tail × K. Sharing the prefix might not
                         pay back the merge launch. PER_BEAM may win.
                         **Adversarial — cost model can over-pick SHARED.**
  tiny_lp                L_p ≤ 256. Definite paged-wins regime.
                         **Adversarial.**
  k_one                  K=1 greedy decode. No sharing benefit.
                         **Adversarial — paged always wins.**
  high_g_small_inter     Many groups (G=K/2 or K/4), tiny intermediate
                         run. 3-level launches G groups but each does
                         minimal work — may lose to 2-level.
  all_unique             G=K (every beam from a unique parent), no
                         intermediate run. Falls back to 2-level.
                         Marginal vs paged at small K.

Per shape we time:
  * paged       — kernel-level equivalent of paged decode.
  * 2-level     — fused cascade with prefix + per-beam tails.
  * 3-level     — fused cascade with prefix + intermediate + tails.
  * bs_kernel   — cost-model pick across {paged, 2-level, 3-level}
                  using the calibrated coefficients.

Reports:
  * best kernel per shape
  * bs_kernel pick agreement (%) per category
  * speedup of bs_kernel pick vs paged per category
  * cells where bs_kernel under-performs (margin > 5%)
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import flashinfer
from flashinfer.testing import bench_gpu_time

from beam_engine.methods.bs_kernel.calibrate import load_or_defaults
from beam_engine.methods.bs_kernel.cost_model import (
    Coefficients,
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


# ---------------------------------------------------------------------------
# Shape spec
# ---------------------------------------------------------------------------


@dataclass
class TreeShape:
    name: str
    K: int
    L_p: int
    inter_len: int        # group-shared intermediate length
    G: int                # number of fork groups
    tail: int             # per-beam unique tail length
    uniform: bool = True  # if False, depth-3 inapplicable
    category: str = "uncategorized"


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Shape families
# ---------------------------------------------------------------------------


def all_shapes(K_values, L_p_values) -> list[TreeShape]:
    out: list[TreeShape] = []

    # Family A: long_prefix_flat (G=1, no intermediate)
    for K in K_values:
        for L_p in L_p_values:
            out.append(TreeShape(
                f"flat                    K={K:<3d} L_p={L_p}",
                K=K, L_p=L_p, inter_len=0, G=1, tail=16,
                category="long_prefix_flat",
            ))

    # Family B: long_prefix_uniform — 3-level should dominate
    for K in K_values:
        for L_p in L_p_values:
            for G in (2, 4, 8):
                if K % G != 0 or G >= K:
                    continue
                out.append(TreeShape(
                    f"uniform G={G:<2d}             K={K:<3d} L_p={L_p}",
                    K=K, L_p=L_p, inter_len=512, G=G, tail=16,
                    category="long_prefix_uniform",
                ))

    # Family C: high_g_small_inter — many groups, tiny intermediate
    for K in K_values:
        for L_p in L_p_values:
            for G_frac in (2, 4):
                G = K // G_frac
                if G < 2 or G >= K or K % G != 0:
                    continue
                out.append(TreeShape(
                    f"high_G G={G:<2d} inter=64    K={K:<3d} L_p={L_p}",
                    K=K, L_p=L_p, inter_len=64, G=G, tail=16,
                    category="high_g_small_inter",
                ))

    # Family D: long_prefix_skewed (uniform=False — 3-level inapplicable)
    for K in K_values:
        for L_p in L_p_values:
            if K < 4:
                continue
            out.append(TreeShape(
                f"skewed 1+singletons     K={K:<3d} L_p={L_p}",
                K=K, L_p=L_p, inter_len=512, G=K - 1,
                tail=16, uniform=False,
                category="long_prefix_skewed",
            ))

    # Family E: short_prefix — adversarial: SHARED may not pay back merge
    for K in K_values:
        if K < 4:
            continue
        for L_p in (256, 512, 1024):
            out.append(TreeShape(
                f"short_prefix            K={K:<3d} L_p={L_p}",
                K=K, L_p=L_p, inter_len=0, G=1, tail=16,
                category="short_prefix",
            ))

    # Family F: tiny_lp — even more adversarial
    for K in K_values:
        if K < 4:
            continue
        for L_p in (64, 128):
            out.append(TreeShape(
                f"tiny_lp                 K={K:<3d} L_p={L_p}",
                K=K, L_p=L_p, inter_len=0, G=1, tail=16,
                category="tiny_lp",
            ))

    # Family G: K=1 greedy — paged should always win
    for L_p in L_p_values:
        out.append(TreeShape(
            f"k_one                   K=1   L_p={L_p}",
            K=1, L_p=L_p, inter_len=0, G=1, tail=16,
            category="k_one",
        ))

    # Family H: all_unique (G=K, no intermediate) — falls to 2-level
    for K in K_values:
        for L_p in L_p_values:
            if K < 4:
                continue
            out.append(TreeShape(
                f"all_unique G=K          K={K:<3d} L_p={L_p}",
                K=K, L_p=L_p, inter_len=0, G=K, tail=16,
                uniform=False,
                category="all_unique",
            ))

    # Family I: boundary_k — K just above CTA_TILE_Q boundaries.
    # Stress-tests the dispatcher's T-choice logic. T_SMALL=16,
    # T_LARGE_CHOICES=(64,128). Padding waste at (K, T):
    #   K=17:  (17/16=2 tiles, 47% pad); (17/64=1, 73% pad); (17/128=1, 87%)
    #   K=33:  (33/16=3, 35%);            (33/64=1, 48%);     (33/128=1, 74%)
    #   K=65:  (65/16=5, 35%);            (65/64=2, 49%);     (65/128=1, 49%)
    #   K=129: (129/16=9, 11%);           (129/64=3, 33%);    (129/128=2, 49%)
    # The right T pick depends on workload: short L_p → tile count
    # dominates → small T loses to launch overhead; long L_p → padding
    # dominates → small T may win.
    for K in (17, 33, 48, 65, 80, 96, 129):
        for L_p in (2048, 8192, 32768):
            out.append(TreeShape(
                f"boundary_K              K={K:<3d} L_p={L_p}",
                K=K, L_p=L_p, inter_len=0, G=1, tail=16,
                category="boundary_k",
            ))

    return out


# ---------------------------------------------------------------------------
# Cascade-input builders
# ---------------------------------------------------------------------------


def _alloc_kv_cache(num_pages: int, device) -> torch.Tensor:
    return torch.randn(
        num_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device,
    )


def _level_groups(group_sizes, tokens, page_cursor, device):
    pages_per_group = _ceil_div(tokens, PAGE_SIZE) if tokens > 0 else 0
    last_page = tokens - (pages_per_group - 1) * PAGE_SIZE if tokens > 0 else 0
    qo_indptr = [0]
    for g in group_sizes:
        qo_indptr.append(qo_indptr[-1] + g)
    kv_indptr = [0]
    kv_indices = []
    kv_last = []
    for _ in group_sizes:
        kv_indices.extend(range(page_cursor, page_cursor + pages_per_group))
        kv_indptr.append(kv_indptr[-1] + pages_per_group)
        kv_last.append(last_page if pages_per_group > 0 else 0)
        page_cursor += pages_per_group
    return (
        torch.tensor(qo_indptr, dtype=torch.int32, device=device),
        torch.tensor(kv_indptr, dtype=torch.int32, device=device),
        torch.tensor(kv_indices, dtype=torch.int32, device=device),
        torch.tensor(kv_last, dtype=torch.int32, device=device),
        page_cursor,
    )


def build_two_level(shape, device):
    K = shape.K
    total_pages = _ceil_div(shape.L_p, PAGE_SIZE) + K * _ceil_div(shape.tail, PAGE_SIZE)
    kv_data = _alloc_kv_cache(total_pages, device)
    page = 0
    levels = []
    for tokens, group_sizes in [
        (shape.L_p, [K]),
        (shape.tail, [1] * K),
    ]:
        out = _level_groups(group_sizes, tokens, page, device)
        levels.append(out[:4])
        page = out[4]
    return kv_data, levels, K


def build_three_level_uniform(shape, device):
    assert shape.uniform and shape.K % shape.G == 0 and shape.inter_len > 0
    K = shape.K
    G = shape.G
    bpg = K // G
    total_pages = (
        _ceil_div(shape.L_p, PAGE_SIZE)
        + G * _ceil_div(shape.inter_len, PAGE_SIZE)
        + K * _ceil_div(shape.tail, PAGE_SIZE)
    )
    kv_data = _alloc_kv_cache(total_pages, device)
    page = 0
    levels = []
    for tokens, group_sizes in [
        (shape.L_p, [K]),
        (shape.inter_len, [bpg] * G),
        (shape.tail, [1] * K),
    ]:
        out = _level_groups(group_sizes, tokens, page, device)
        levels.append(out[:4])
        page = out[4]
    return kv_data, levels, K


# ---------------------------------------------------------------------------
# Per-kernel runs
# ---------------------------------------------------------------------------


def _bench(fn) -> float:
    times = bench_gpu_time(fn, dry_run_iters=5, repeat_iters=30, cold_l2_cache=False)
    return statistics.median(times) * 1000.0  # ms → µs


def run_two_level(shape, device, *, force_t: int | None = None) -> float:
    if shape.K < 1:
        return float("nan")
    try:
        kv_data, levels, K = build_two_level(shape, device)
        qo_arr, kvp_arr, kvi_arr, kvl_arr = (list(x) for x in zip(*levels))
        q = torch.randn(K, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        if K == 1 or shape.L_p == 0:
            return float("nan")
        w = flashinfer.FusedMultiLevelCascadeAttentionWrapper(
            num_levels=2, kv_layout="NHD",
        )
        plan_kwargs = {}
        if force_t is not None:
            plan_kwargs["force_cta_tile_q"] = force_t
        w.plan(qo_arr, kvp_arr, kvi_arr, kvl_arr,
               NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
               **plan_kwargs)
        return _bench(lambda: w.run(q, kv_data))
    except Exception:
        return float("nan")


def run_three_level(shape, device) -> float:
    if not shape.uniform or shape.inter_len == 0 or shape.K % shape.G != 0:
        return float("nan")
    if shape.G < 2 or shape.G >= shape.K:
        return float("nan")
    try:
        kv_data, levels, K = build_three_level_uniform(shape, device)
        qo_arr, kvp_arr, kvi_arr, kvl_arr = (list(x) for x in zip(*levels))
        q = torch.randn(K, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        w = flashinfer.FusedMultiLevelCascadeAttentionWrapper(
            num_levels=3, kv_layout="NHD",
        )
        w.plan(qo_arr, kvp_arr, kvi_arr, kvl_arr,
               NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE)
        return _bench(lambda: w.run(q, kv_data))
    except Exception:
        return float("nan")


def run_tree_fused(shape, device) -> float:
    """Tree-attention via BatchPrefillWithRaggedKVCacheWrapper + custom
    mask (the kernel arXiv:2502.00085 / `baselines/tree.py` uses).

    Single request with qo_len=K, kv_len=total_kv (fused buffer).
    Custom mask encodes per-beam visibility — each query i sees the
    shared prefix, its fork-group's intermediate run (if any), and its
    own per-beam tail.
    """
    K = shape.K
    if K < 1:
        return float("nan")
    inter = shape.inter_len if shape.uniform and shape.G > 1 and shape.inter_len > 0 else 0
    G = shape.G if inter > 0 else 1
    if G > 1 and K % G != 0:
        # Non-uniform tree-mask construction — skip for now.
        return float("nan")
    bpg = K // G if G > 1 else K

    kv_total = shape.L_p + G * inter + K * shape.tail
    if kv_total == 0:
        return float("nan")

    try:
        k_buf = torch.randn(kv_total, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        v_buf = torch.randn(kv_total, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)

        # Build mask [K, kv_total].
        mask = torch.zeros(K, kv_total, dtype=torch.bool, device=device)
        # All beams see the shared prefix.
        if shape.L_p > 0:
            mask[:, : shape.L_p] = True
        # Group-shared intermediate (if depth-3 uniform).
        if G > 1 and inter > 0:
            for g in range(G):
                inter_start = shape.L_p + g * inter
                inter_end = inter_start + inter
                mask[g * bpg : (g + 1) * bpg, inter_start:inter_end] = True
        # Per-beam tails.
        tail_offset = shape.L_p + G * inter
        for i in range(K):
            mask[i, tail_offset + i * shape.tail : tail_offset + (i + 1) * shape.tail] = True

        qo_indptr = torch.tensor([0, K], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor([0, kv_total], dtype=torch.int32, device=device)
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
        w.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=NUM_QO_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim_qk=HEAD_DIM,
            causal=False,
            custom_mask=mask.flatten(),
        )
        q = torch.randn(K, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        return _bench(lambda: w.run(q, k_buf, v_buf))
    except Exception:
        return float("nan")


def run_per_beam(shape, device) -> float:
    """Paged-equivalent: K independent sequences, each scanning the
    entire prefix + intermediate + tail."""
    K = shape.K
    full_len = shape.L_p + shape.inter_len + shape.tail
    if full_len == 0:
        return float("nan")
    pages_per_seq = _ceil_div(full_len, PAGE_SIZE)
    last_page = full_len - (pages_per_seq - 1) * PAGE_SIZE
    kv_data = _alloc_kv_cache(pages_per_seq, device)

    qo_indptr = torch.arange(K + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.tensor(
        [i * pages_per_seq for i in range(K + 1)],
        dtype=torch.int32, device=device,
    )
    kv_indices = torch.tensor(
        list(range(pages_per_seq)) * K, dtype=torch.int32, device=device,
    )
    kv_last = torch.full((K,), last_page, dtype=torch.int32, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    w.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=kv_indptr,
        paged_kv_indices=kv_indices,
        paged_kv_last_page_len=kv_last,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        page_size=PAGE_SIZE,
        causal=False,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    q = torch.randn(K, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    return _bench(lambda: w.run(q, kv_data))


# ---------------------------------------------------------------------------
# Cost-model pick
# ---------------------------------------------------------------------------


def pick_for_shape(shape: TreeShape, coeffs: Coefficients) -> Strategy:
    intermediate = (
        IntermediateShape(G=shape.G, group_size=shape.K // shape.G,
                          inter_len_tokens=shape.inter_len)
        if shape.uniform and shape.inter_len > 0 and shape.K % shape.G == 0 and 1 < shape.G < shape.K
        else None
    )
    w = WorkloadShape(
        K=shape.K,
        L_p=shape.L_p,
        suffix_lens=[shape.tail] * shape.K,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        bytes_per_kv=2 * NUM_KV_HEADS * HEAD_DIM * 2,
        intermediate=intermediate,
    )
    return pick_strategy(w, coeffs).strategy


def strategy_to_kernel(s: Strategy) -> str:
    if s == Strategy.PER_BEAM:
        return "paged"
    if s.value.startswith("shared_3l"):
        return "3-level"
    return "2-level"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--L_p", nargs="+", type=int, default=[2048, 8192, 32768, 65536])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    coeffs = load_or_defaults(device)

    shapes = all_shapes(args.K, args.L_p)
    print(f"benchmarking {len(shapes)} tree shapes\n")

    rows = []
    print(f"{'category':22s}  {'shape':45s}  "
          f"{'paged':>9s}  {'tree':>9s}  {'2level':>9s}  {'3level':>9s}  "
          f"{'best':>10s}  {'pick':>8s}  {'reg%':>5s}")
    print("-" * 140)

    for s in shapes:
        t_paged = run_per_beam(s, device)
        t_tree = run_tree_fused(s, device)
        t_2l = run_two_level(s, device)
        t_3l = run_three_level(s, device)

        candidates = {
            "paged": t_paged,
            "tree": t_tree,
            "2-level": t_2l,
            "3-level": t_3l,
        }
        valid = {k: v for k, v in candidates.items() if v == v}
        if not valid:
            continue
        best_kernel = min(valid, key=valid.get)
        best_us = valid[best_kernel]

        pick = pick_for_shape(s, coeffs)
        pick_kernel = strategy_to_kernel(pick)

        # Regret of bs_kernel pick.
        if pick_kernel in valid:
            pick_us = valid[pick_kernel]
            regret = (pick_us / best_us - 1.0) * 100.0
        else:
            pick_us = float("nan")
            regret = float("nan")

        print(
            f"{s.category:22s}  {s.name:45s}  "
            f"{t_paged:9.1f}  {t_tree:9.1f}  {t_2l:9.1f}  {t_3l:9.1f}  "
            f"{best_kernel:>10s}  {pick_kernel:>8s}  {regret:>5.1f}"
        )
        rows.append({
            "category": s.category, "shape": s.name,
            "K": s.K, "L_p": s.L_p, "G": s.G,
            "inter_len": s.inter_len, "tail": s.tail, "uniform": s.uniform,
            "paged_us": t_paged, "tree_us": t_tree,
            "two_level_us": t_2l, "three_level_us": t_3l,
            "best_kernel": best_kernel, "best_us": best_us,
            "pick_kernel": pick_kernel, "pick_us": pick_us,
            "regret_pct": regret,
        })

    # ---------- Per-category summary ----------
    print("\n" + "=" * 90)
    print(f"{'category':22s}  {'n':>3s}  {'pick=best':>10s}  "
          f"{'mean_regret':>12s}  {'max_regret':>11s}  "
          f"{'mean_speedup_vs_paged':>22s}")
    print("-" * 90)

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    for cat, rs in sorted(by_cat.items()):
        n = len(rs)
        n_correct = sum(1 for r in rs if r["pick_kernel"] == r["best_kernel"])
        regrets = [r["regret_pct"] for r in rs if r["regret_pct"] == r["regret_pct"]]
        mean_regret = statistics.mean(regrets) if regrets else float("nan")
        max_regret = max(regrets) if regrets else float("nan")
        # Speedup vs paged for the bs_kernel pick.
        speedups = []
        for r in rs:
            if r["paged_us"] == r["paged_us"] and r["pick_us"] == r["pick_us"] and r["pick_us"] > 0:
                speedups.append(r["paged_us"] / r["pick_us"])
        mean_sp = statistics.mean(speedups) if speedups else float("nan")
        print(
            f"{cat:22s}  {n:>3d}  {n_correct}/{n:<8d}  "
            f"{mean_regret:>11.2f}%  {max_regret:>10.2f}%  "
            f"{mean_sp:>21.2f}×"
        )

    # ---------- Per-baseline speedup table ----------
    print("\n" + "=" * 100)
    print("Mean speedup of bs_kernel pick vs each baseline kernel, by category:")
    print(f"{'category':22s}  {'vs_paged':>10s}  {'vs_tree':>10s}  "
          f"{'vs_2level':>10s}  {'vs_3level':>10s}")
    print("-" * 80)
    for cat, rs in sorted(by_cat.items()):
        sp_paged, sp_tree, sp_2l, sp_3l = [], [], [], []
        for r in rs:
            pick = r["pick_us"]
            if pick != pick or pick <= 0:
                continue
            if r["paged_us"] == r["paged_us"]:
                sp_paged.append(r["paged_us"] / pick)
            if r["tree_us"] == r["tree_us"]:
                sp_tree.append(r["tree_us"] / pick)
            if r["two_level_us"] == r["two_level_us"]:
                sp_2l.append(r["two_level_us"] / pick)
            if r["three_level_us"] == r["three_level_us"]:
                sp_3l.append(r["three_level_us"] / pick)

        def fmt(xs):
            return f"{statistics.mean(xs):8.2f}×" if xs else "       —"
        print(f"{cat:22s}  {fmt(sp_paged):>10s}  {fmt(sp_tree):>10s}  "
              f"{fmt(sp_2l):>10s}  {fmt(sp_3l):>10s}")

    # ---------- Regret outliers ----------
    bad = [r for r in rows if r["regret_pct"] == r["regret_pct"] and r["regret_pct"] > 5.0]
    if bad:
        print("\nUnder-performance (bs_kernel regret > 5%):")
        for r in sorted(bad, key=lambda x: -x["regret_pct"]):
            print(
                f"  [{r['category']}] {r['shape']:45s}  "
                f"pick={r['pick_kernel']:>7s} ({r['pick_us']:.1f}us)  "
                f"best={r['best_kernel']:>7s} ({r['best_us']:.1f}us)  "
                f"regret={r['regret_pct']:.1f}%"
            )
    else:
        print("\n(no shapes with regret > 5%)")

    # ---------- CTA_Q padding T-sensitivity ----------
    boundary_shapes = [s for s in shapes if s.category == "boundary_k"]
    if boundary_shapes:
        print("\n" + "=" * 100)
        print("CTA_Q padding sensitivity (boundary_k shapes — K just above tile boundaries):")
        print(f"  {'shape':40s}  {'T=16':>8s}  {'T=64':>8s}  {'T=128':>8s}  "
              f"{'best_T':>8s}  {'pick_T':>7s}  {'reg%':>5s}")
        print("-" * 100)
        for s in boundary_shapes:
            t_16 = run_two_level(s, device, force_t=16)
            t_64 = run_two_level(s, device, force_t=64)
            t_128 = run_two_level(s, device, force_t=128)
            t_table = {"16": t_16, "64": t_64, "128": t_128}
            valid = {k: v for k, v in t_table.items() if v == v}
            if not valid:
                continue
            best_T = min(valid, key=valid.get)
            best_us = valid[best_T]
            # Cost-model pick (T_large the model would force).
            from beam_engine.methods.bs_kernel.cost_model import T_LARGE_CHOICES
            wsh = WorkloadShape(
                K=s.K, L_p=s.L_p, suffix_lens=[s.tail] * s.K,
                num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
                bytes_per_kv=2 * NUM_KV_HEADS * HEAD_DIM * 2,
                intermediate=None,
            )
            pick = pick_strategy(wsh, coeffs)
            if pick.strategy == Strategy.PER_BEAM:
                pick_T_str = "paged"
                pick_us = float("nan")
            else:
                pick_T_str = str(pick.t_large) if pick.pool_count == 1 else "auto"
                # When pool_count=2, the wrapper auto-selects; just report pick.t_large.
                pick_us = valid.get(str(pick.t_large), float("nan"))
            if pick_us == pick_us:
                regret = (pick_us / best_us - 1.0) * 100.0
            else:
                regret = float("nan")
            print(f"  {s.name:40s}  {t_16:8.1f}  {t_64:8.1f}  {t_128:8.1f}  "
                  f"{best_T:>8s}  {pick_T_str:>7s}  {regret:>5.1f}")

    # ---------- CSV ----------
    if args.out is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        gpu = torch.cuda.get_device_name().replace(" ", "_")
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.out = str(results_dir / f"tree_shapes-{gpu}-{ts}.csv")
    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

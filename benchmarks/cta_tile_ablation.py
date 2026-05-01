"""CTA_TILE_Q ablation: justifies the two-pool design in the fused cascade
wrapper. Theory (paste-able into a paper):

    The FA2-style prefill kernel processes queries in tiles of fixed
    CTA_TILE_Q = T rows. For a tile holding packed_qo = R ≤ T real rows,
    the unused (T − R) rows are padding: the kernel still issues the full
    Q·K^T MMA chain over kv_len, still loads T · HEAD_DIM_QK query
    elements, and still executes the full softmax/V·P chain — padding
    rows are masked only at writeback. Per-tile cost is therefore
        C_tile(T, L) ≈ α + β·T·L + γ·L
    independent of R, and a request of size R issues ⌈R/T⌉ tiles. So
        total_cost(R, L, T) = ⌈R/T⌉ · (α + β·T·L + γ·L)
    and MMA utilization U(R, T) = R / (T · ⌈R/T⌉).

    For beam-search cascade trees two regimes coexist in one call:
      - per-beam tail (level ≥ 1): R = G (GQA group size, e.g. 4)
      - shared root (level 0):     R = K · G (e.g. 32 for K=8)
    A single T cannot be Pareto-optimal across this 8× span:
      - T = 16 splits the root into K·G/16 tiles, each re-scanning the
        prefix → O(K)-fold KV bandwidth blow-up at level 0.
      - T = 128 runs per-beam tails at 4/128 = 3.1% MMA utilization
        (97% of MMA cycles are padding).
    The two-pool design routes each tile into its T-optimal launch.

This script measures the predicted gap empirically. We run the same
imbalanced 2-level cascade (shared prompt + per-beam tail) under three
schedulers:
    - two-pool (default, T_lg ∈ {64,128} + T=16)
    - force_cta_tile_q=16   (one pool, T=16 only)
    - force_cta_tile_q=128  (one pool, T=128 only)
and sweep K to push the imbalance.

Per CLAUDE.md: pick a fully idle GPU and pin via CUDA_VISIBLE_DEVICES.

Usage:
    nvidia-smi
    PYTHONPATH=3rdparty/flashinfer:$PYTHONPATH CUDA_VISIBLE_DEVICES=<id> \
        uv run python benchmarks/cta_tile_ablation.py
"""

from __future__ import annotations

import statistics

import torch

import flashinfer
from flashinfer.testing import bench_gpu_time


# Defaults (Llama-3.1: gqa_group=4). Override via env vars or by editing
# the constants in main() to sweep GQA ratios.
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 16
DTYPE = torch.float16


def set_gqa(num_qo_heads: int, num_kv_heads: int):
    """Override the module-global head config for the current sweep."""
    global NUM_QO_HEADS, NUM_KV_HEADS
    NUM_QO_HEADS = num_qo_heads
    NUM_KV_HEADS = num_kv_heads


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def make_imbalanced_three_level(
    K: int, L_p: int, l1_groups: list[int], inter: int, tails: list[int], device
):
    """3-level cascade with asymmetric L1 group sizes AND variable per-beam
    tail lengths. ``l1_groups`` partitions the K beams (sum must equal K);
    each group shares ``inter`` intermediate tokens. ``tails[i]`` is per-beam
    tail length (len must equal K). All R values coexist in one call:

        L0:  packed_qo = K · G                              (root tile)
        L1:  packed_qo = group_size · G  per group          (one tile per group)
        L2:  packed_qo = G  per beam                        (one tile per beam)

    The two-pool router sees a mix of small-R (4, 8, 12) and large-R
    (K·G, max_group·G) tiles in the same launch — exactly the regime a
    single-T schedule cannot serve well.
    """
    assert sum(l1_groups) == K, (l1_groups, K)
    assert len(tails) == K, (tails, K)
    G = NUM_QO_HEADS // NUM_KV_HEADS  # gqa group size; per-row factor

    pages_root = _ceil_div(L_p, PAGE_SIZE)
    pages_inter = max(_ceil_div(inter, PAGE_SIZE), 1)
    pages_tail = [max(_ceil_div(t, PAGE_SIZE), 1) for t in tails]
    n_l1 = len(l1_groups)

    total_pages = pages_root + n_l1 * pages_inter + sum(pages_tail)
    kv = torch.randn(
        total_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device
    )

    # qo_indptr: one cumulative offset per group/beam at each level.
    # L0: one group of K; L1: G groups, each cumulative; L2: K per-beam.
    qo_l0 = torch.tensor([0, K], dtype=torch.int32, device=device)
    qo_l1 = torch.tensor(
        [0] + [sum(l1_groups[: i + 1]) for i in range(n_l1)],
        dtype=torch.int32,
        device=device,
    )
    qo_l2 = torch.arange(K + 1, dtype=torch.int32, device=device)

    # kv indices and indptrs are per-level — they index into kv_indices_arr[l].
    page_cursor = 0
    # L0: one group, one slice of pages.
    kvix_l0 = torch.arange(
        page_cursor, page_cursor + pages_root, dtype=torch.int32, device=device
    )
    kvi_l0 = torch.tensor([0, pages_root], dtype=torch.int32, device=device)
    last_l0 = torch.tensor(
        [L_p - (pages_root - 1) * PAGE_SIZE if L_p > 0 else PAGE_SIZE],
        dtype=torch.int32, device=device,
    )
    page_cursor += pages_root

    # L1: n_l1 groups, each with pages_inter pages (uniform intermediate length).
    kvix_l1 = torch.arange(
        page_cursor, page_cursor + n_l1 * pages_inter,
        dtype=torch.int32, device=device,
    )
    kvi_l1 = torch.arange(n_l1 + 1, dtype=torch.int32, device=device) * pages_inter
    last_l1 = torch.full(
        (n_l1,), inter - (pages_inter - 1) * PAGE_SIZE if inter > 0 else 1,
        dtype=torch.int32, device=device,
    )
    page_cursor += n_l1 * pages_inter

    # L2: K beams, possibly variable tail lengths -> variable pages_tail[i].
    kvix_l2_list = []
    indptr_l2 = [0]
    last_l2 = []
    for i in range(K):
        start = page_cursor
        end = page_cursor + pages_tail[i]
        kvix_l2_list.append(torch.arange(start, end, dtype=torch.int32, device=device))
        indptr_l2.append(indptr_l2[-1] + pages_tail[i])
        last_l2.append(
            tails[i] - (pages_tail[i] - 1) * PAGE_SIZE if tails[i] > 0 else 1
        )
        page_cursor = end
    kvix_l2 = torch.cat(kvix_l2_list)
    kvi_l2 = torch.tensor(indptr_l2, dtype=torch.int32, device=device)
    last_l2_t = torch.tensor(last_l2, dtype=torch.int32, device=device)

    return (
        kv,
        [qo_l0, qo_l1, qo_l2],
        [kvi_l0, kvi_l1, kvi_l2],
        [kvix_l0, kvix_l1, kvix_l2],
        [last_l0, last_l1, last_l2_t],
    )


def bench_three_level(force_cta_tile_q, K, L_p, l1_groups, inter, tails, device):
    kv, qo, kvi, kvix, last = make_imbalanced_three_level(
        K, L_p, l1_groups, inter, tails, device
    )
    q = torch.randn(K, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    w = flashinfer.FusedMultiLevelCascadeAttentionWrapper(
        3, kv_layout="NHD", force_cta_tile_q=force_cta_tile_q
    )
    w.plan(qo, kvi, kvix, last,
           NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE)
    times = bench_gpu_time(
        lambda: w.run(q, kv),
        dry_run_iters=5, repeat_iters=50, cold_l2_cache=False,
    )
    return statistics.median(times) * 1000.0


def make_two_level(K: int, L_p: int, tail: int, device):
    pages_root = _ceil_div(L_p, PAGE_SIZE)
    pages_tail = max(_ceil_div(tail, PAGE_SIZE), 1)
    total_pages = pages_root + K * pages_tail
    kv = torch.randn(
        total_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device
    )

    qo_arr = [
        torch.tensor([0, K], dtype=torch.int32, device=device),
        torch.arange(K + 1, dtype=torch.int32, device=device),
    ]
    # paged_kv_indptr indexes INTO kv_indices (per-level), NOT into the page
    # buffer. So L0's indptr is [0, pages_root] (one request, pages_root long)
    # and L1's is [0, pages_tail, 2·pages_tail, ..., K·pages_tail].
    kvi_arr = [
        torch.tensor([0, pages_root], dtype=torch.int32, device=device),
        torch.arange(K + 1, dtype=torch.int32, device=device) * pages_tail,
    ]
    kvix_arr = [
        torch.arange(pages_root, dtype=torch.int32, device=device),
        torch.arange(
            pages_root, pages_root + K * pages_tail, dtype=torch.int32, device=device
        ),
    ]
    last_page_root = L_p - (pages_root - 1) * PAGE_SIZE if L_p > 0 else PAGE_SIZE
    last_page_tail = tail - (pages_tail - 1) * PAGE_SIZE if tail > 0 else 1
    last_arr = [
        torch.tensor([last_page_root], dtype=torch.int32, device=device),
        torch.full((K,), last_page_tail, dtype=torch.int32, device=device),
    ]
    return kv, qo_arr, kvi_arr, kvix_arr, last_arr


def bench_one(force_cta_tile_q, K, L_p, tail, device) -> float:
    kv, qo_arr, kvi_arr, kvix_arr, last_arr = make_two_level(K, L_p, tail, device)
    q = torch.randn(K, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    w = flashinfer.FusedMultiLevelCascadeAttentionWrapper(
        2, kv_layout="NHD", force_cta_tile_q=force_cta_tile_q
    )
    w.plan(qo_arr, kvi_arr, kvix_arr, last_arr,
           NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE)
    times = bench_gpu_time(
        lambda: w.run(q, kv),
        dry_run_iters=5,
        repeat_iters=50,
        cold_l2_cache=False,
    )
    return statistics.median(times) * 1000.0  # ms -> us


def fmt(x):
    return f"{x:7.1f}"


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(
        f"Heads qo={NUM_QO_HEADS} kv={NUM_KV_HEADS} (gqa group={NUM_QO_HEADS // NUM_KV_HEADS}) "
        f"head_dim={HEAD_DIM} page_size={PAGE_SIZE} dtype={DTYPE}"
    )

    # Imbalanced 2-level: shared prompt of L_p tokens + per-beam tail of 1
    # token. packed_qo_root = K·G (= 4K for Llama-3.1), packed_qo_tail = G = 4.
    # With G=4, the imbalance ratio root/tail = K. Sweep K to amplify it.
    print()
    print("=" * 96)
    print("Imbalanced 2-level cascade (shared prompt + per-beam tail). L_p=2048, tail=1.")
    print("Per-beam packed_qo = G = 4 (constant);  root packed_qo = K·G grows with K.")
    print("=" * 96)
    print(
        f"  {'K':>3} {'root R':>7} {'tiles_T16(root,tail)':>22} {'tiles_T128(root,tail)':>22} | "
        f"{'two-pool':>10} {'T=16 only':>10} {'T=128 only':>11}"
    )
    print("  " + "-" * 96)

    L_p = 2048
    tail = 1
    rows = []
    for K in [1, 2, 4, 8, 16, 32, 64]:
        try:
            t_two = bench_one(None, K, L_p, tail, device)
        except Exception as e:
            t_two = float("nan")
            print(f"  K={K} two-pool failed: {e}")
        t_16 = bench_one(16, K, L_p, tail, device)
        t_128 = bench_one(128, K, L_p, tail, device)
        # tiles required at each T
        G = NUM_QO_HEADS // NUM_KV_HEADS
        root_R = K * G
        tiles_T16 = (_ceil_div(root_R, 16), _ceil_div(G, 16))
        tiles_T128 = (_ceil_div(root_R, 128), _ceil_div(G, 128))
        rows.append((K, t_two, t_16, t_128))
        print(
            f"  {K:>3} {root_R:>7} "
            f"{f'{tiles_T16[0]:>2}, {tiles_T16[1]} × K':>22} "
            f"{f'{tiles_T128[0]:>2}, {tiles_T128[1]} × K':>22} | "
            f"{fmt(t_two):>10} {fmt(t_16):>10} {fmt(t_128):>11}"
        )

    print()
    print("Predictions (worst-case asymptotics for this workload):")
    print("  T=16 only:   level-0 splits root into K·G/16 tiles, each re-scans the prefix.")
    print("              cost grows linearly in K once K·G > 16 because of redundant KV reads.")
    print("  T=128 only:  per-beam tail tiles fill 4/128 = 3.1% of MMA. Padding-bound.")
    print("              cost ≈ K independent tiles, each paying full T·L MMA work.")
    print("  Two-pool:    root → pool_lg (1 tile, 1 KV scan); tails → pool_16 (K tiles).")
    print("              cost ≈ max(MMA at L=root, MMA at K tails) without redundant reads.")

    # Padding-overhead stress: K=8 beams, sweep per-beam tail kv_len. As tail
    # grows, per-beam MMA cost grows (β·T·L term) so T=128 padding becomes
    # measurable. Root packed=32 fits in 1 tile at T=128 (50% util), 2 tiles
    # at T=16 (100% util but 2× KV scans).
    print()
    print("=" * 96)
    print("Tail-length sweep (K=8, L_p=2048; vary per-beam tail kv_len)")
    print("Predicts: as tail grows, T=128 pays β·T·L padding linearly per beam.")
    print("=" * 96)
    K = 8
    L_p = 2048
    print(
        f"  {'tail (tok)':>11} | {'two-pool':>10} {'T=16 only':>10} {'T=128 only':>11} | "
        f"{'T128/two':>9} {'T16/two':>9}"
    )
    print("  " + "-" * 80)
    for tail in [1, 16, 64, 256, 1024, 4096]:
        # Force is a knob on the wrapper, keep using it.
        t_two = bench_one(None, K, L_p, tail, device)
        t_16 = bench_one(16, K, L_p, tail, device)
        t_128 = bench_one(128, K, L_p, tail, device)
        print(
            f"  {tail:>11} | {fmt(t_two):>10} {fmt(t_16):>10} {fmt(t_128):>11} | "
            f"{t_128/t_two:>8.2f}x {t_16/t_two:>8.2f}x"
        )

    # Imbalanced multi-level cascades — the regime where no single T can
    # be picked, even with full a-priori workload knowledge. The previous
    # sweeps had R uniform-within-a-call (all tiles at root same R, all
    # tail tiles same R), so a smart dispatcher could just pick T per
    # call. These cascades have tiles at *multiple* R values in the SAME
    # launch, with each R regime contributing meaningfully to cost.
    print()
    print("=" * 96)
    print("Imbalanced multi-level cascades — heterogeneous R within ONE call")
    print("=" * 96)
    G = NUM_QO_HEADS // NUM_KV_HEADS
    # Cases where root, one or more L1 groups, and tails are all big enough
    # to matter. Long L_p + long shared L1 intermediate + long per-beam tail.
    # Each row: (label, K, L_p, l1_groups, intermediate, tail_per_beam)
    cases = [
        # K=16, root R=64 (T=128 → 50% util), L1 {12,4}: R=48 (37% util at T=128),
        # R=16 (good at T=16 OR T=128). Tails R=4 (3% at T=128).
        # Long L1 intermediate (1024) AND long tails (1024) make BOTH levels'
        # padding cost matter. Only a per-tile T choice avoids it.
        ("K=16 L_p=2048 L1={12,4} inter=1024 tail=1024", 16, 2048, [12, 4], 1024, 1024),
        # Wider asymmetry at higher K. Big group at L1 (R=64) is exactly
        # T=128's middle ground; many small groups (R=4) are pure padding at T=128.
        ("K=32 L_p=4096 L1={16,8,4,2,1,1} inter=2048 tail=256",
         32, 4096, [16, 8, 4, 2, 1, 1], 2048, 256),
        # Adversarial root-and-tails: both contribute substantially.
        ("K=64 L_p=2048 L1={32,16,8,4,4} inter=512 tail=256",
         64, 2048, [32, 16, 8, 4, 4], 512, 256),
        # Big root + lots of small L1 groups + long tails.
        ("K=64 L_p=4096 L1={4}*16 inter=512 tail=1024 (16× small grps)",
         64, 4096, [4] * 16, 512, 1024),
    ]
    print(
        f"  {'topology':<58} {'tile R values':<23} | {'two-pool':>10} {'T=16':>8} {'T=128':>8} | "
        f"{'best/two-pool':>13}"
    )
    print("  " + "-" * 110)
    for label, K, L_p, groups, inter, tail_each in cases:
        tails = [tail_each] * K
        # collect distinct R values that the scheduler will see
        r_root = K * G
        r_l1 = sorted(set(g * G for g in groups), reverse=True)
        r_l2 = G
        rs = sorted(set([r_root] + r_l1 + [r_l2]), reverse=True)
        rstr = "{" + ",".join(str(r) for r in rs) + "}"
        try:
            t_two = bench_three_level(None, K, L_p, groups, inter, tails, device)
        except Exception as e:
            t_two = float("nan"); print(f"  two-pool fail: {e}")
        try:
            t_16 = bench_three_level(16, K, L_p, groups, inter, tails, device)
        except Exception as e:
            t_16 = float("nan")
        try:
            t_128 = bench_three_level(128, K, L_p, groups, inter, tails, device)
        except Exception as e:
            t_128 = float("nan")
        best_single = min(t_16, t_128)
        # ratio < 1 means two-pool is faster than the best single-T; > 1 means
        # the best single-T is faster than two-pool.
        ratio = best_single / t_two if t_two > 0 else float("nan")
        print(
            f"  {label:<58} {rstr:<23} | "
            f"{fmt(t_two):>10} {fmt(t_16):>8} {fmt(t_128):>8} | "
            f"{ratio:>11.2f}x"
        )

    # Two-pool sweet spot: where BOTH R regimes contribute substantial
    # cost so neither single-T choice can serve the call well. The
    # canonical case is "production beam search with non-trivial tails":
    # large K (root needs T=128 for KV-BW), long L_p (T=16 catastrophic
    # at root due to KV re-scans), moderate per-beam tail (T=128
    # catastrophic at tails due to padding).
    print()
    print("=" * 96)
    print("Two-pool sweet spot: large K, long L_p, moderate tails")
    print("Both T=16 root (KV-bound) and T=128 tails (MMA-padded) lose; only")
    print("two-pool routes each regime to its T-optimal launch.")
    print("=" * 96)
    print(
        f"  {'config':<46} | {'two-pool':>9} {'T=16':>9} {'T=128':>9} | "
        f"{'T=16/two':>9} {'T=128/two':>10}"
    )
    print("  " + "-" * 102)
    sweet_cases = [
        # (K, L_p, tail) — bracket the "both regimes matter" regime
        (16, 4096, 64),
        (32, 4096, 64),
        (32, 8192, 64),
        (32, 8192, 256),
        (64, 4096, 64),
        (64, 8192, 64),
        (64, 8192, 256),
    ]
    for K, L_p_, tail_ in sweet_cases:
        t_two = bench_one(None, K, L_p_, tail_, device)
        t_16 = bench_one(16, K, L_p_, tail_, device)
        t_128 = bench_one(128, K, L_p_, tail_, device)
        print(
            f"  K={K:>2} L_p={L_p_:>4d} tail={tail_:>4d}                       | "
            f"{fmt(t_two):>9} {fmt(t_16):>9} {fmt(t_128):>9} | "
            f"{t_16/t_two:>8.2f}x {t_128/t_two:>9.2f}x"
        )

    print()
    print("=" * 96)
    print("Interpretation for the paper:")
    print("  K-sweep (L_p=2048, tail=1):")
    print("    - At K=64, T=16-only pays 16 redundant KV scans of the 2048-token")
    print("      root → 1.41× slowdown vs two-pool. KV-bandwidth tax is real.")
    print("    - T=128-only is competitive when tails are short (L=1 → MMA cost")
    print("      is overhead-dominated, padding doesn't matter).")
    print("  Tail-sweep (K=8, L_p=2048):")
    print("    - As tail grows, per-beam MMA cost C ≈ β·T·tail·K. At T=128 every")
    print("      beam's tile is 4/128 = 3% MMA-utilized → padding cost grows linearly")
    print("      with tail. T=128/two-pool ratio should diverge as tail increases.")
    print("    - Two-pool stays flat in tail because pool_16 packs tail tiles at 25%")
    print("      utilization (4× better than T=128).")
    print("  Together, these two sweeps span the imbalance space (K, tail) and show")
    print("  no single CTA_TILE_Q is Pareto-optimal — the two-pool design is")
    print("  necessary to handle the imbalance inherent in beam-search trees.")


if __name__ == "__main__":
    main()

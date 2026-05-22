"""Probe the marginal cost of adding a thin-Q cascade level (L1) to a
``FusedMultiLevelCascadeAttentionWrapper`` invocation.

For each (K, L_p, L_tail) shape:

  T_num2 = wall time of cascade prefill with num_levels=2
           L0: 1 group × K queries × L_p kv tokens   (dense root)
           L1: K groups × 1 query × L_tail kv tokens (thin-Q tail)
  T_num1 = wall time of cascade prefill with num_levels=1
           L0: 1 group × K queries × L_p kv tokens   (root only)

The difference T_num2 - T_num1 is the kernel's actual marginal cost
of processing the thin-Q L1 cascade level.

Compare to the current cost-model prediction
(``per_tile_per_kv_us[64] × cta_tile_kv × n_sub_tiles``) which folds
L1 into L0's wave pool and predicts ~1 µs of marginal cost.

If actual L1 cost >> predicted, the cost model is undercharging 1POOL
at the cascade-tail level — which is the hypothesized cause of the
70B C3 picker mispick (1.19× of oracle, picks 1POOL when DT is 19%
faster on multi_doc_qa K=32 B=1 mn=256).

Usage:
    uv run python benchmarks/bs_kernel/probe_cascade_thin_q.py
"""

from __future__ import annotations

import argparse
import os
import time
import torch
from flashinfer import FusedMultiLevelCascadeAttentionWrapper


def _time_us(fn, n: int = 20, warmup: int = 5, device=None) -> float:
    """Median wall time of ``fn`` over ``n`` runs (µs)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(n):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def probe(
    device: torch.device,
    *,
    K: int, L_p: int, L_tail: int,
    num_kv_heads: int = 4, num_qo_heads: int = 64,
    head_dim: int = 128, page_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    force_l0_chunks: int | None = None,
) -> dict:
    """Return {T_num2_us, T_num1_us, L1_marginal_us} for one shape.

    If ``force_l0_chunks`` is given, sets FLASHINFER_FORCE_KV_CHUNKS=0:N
    so L0's split-K num_chunks is pinned to N in BOTH wrapper1 and
    wrapper2. Used to test whether L0's split-K differs between
    num_levels=1 vs num_levels=2 plans.
    """
    if force_l0_chunks is not None and force_l0_chunks > 0:
        os.environ["FLASHINFER_FORCE_KV_CHUNKS"] = f"0:{force_l0_chunks}"
    else:
        os.environ.pop("FLASHINFER_FORCE_KV_CHUNKS", None)
    # CTA_TILE_Q derived from L0's packed-Q size so L0's natural pool
    # choice is preserved; L1's thin Q's are forced into the *same* pool
    # (over-padded) instead of spawning a separate pool_16 launch. This
    # is the "force same CTA_Q as L0" experiment: it isolates the L1
    # marginal cost from the cross-pool launch tax.
    gqa_group_size = num_qo_heads // num_kv_heads
    packed_L0 = K * gqa_group_size
    if packed_L0 <= 16:
        force_T = 16
    elif packed_L0 <= 64:
        force_T = 64
    else:
        force_T = 128

    # Total pages we need: prefix (shared) + K tails (one slab per beam).
    n_pages_prefix = (L_p + page_size - 1) // page_size
    n_pages_tail_each = (L_tail + page_size - 1) // page_size
    total_pages = n_pages_prefix + K * n_pages_tail_each
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    kv = torch.randn(total_pages, 2, page_size, num_kv_heads, head_dim,
                     dtype=dtype, device=device)

    # ---- num_levels=2 plan: L0 (root) + L1 (per-beam tail) ----
    # Level 0 (root): one group of K queries attending L_p tokens.
    qo_indptr_L0 = torch.tensor([0, K], dtype=torch.int32, device=device)
    kvp_indptr_L0 = torch.tensor([0, n_pages_prefix], dtype=torch.int32, device=device)
    kvi_L0 = torch.arange(n_pages_prefix, dtype=torch.int32, device=device)
    kvl_L0 = torch.tensor(
        [((L_p - 1) % page_size) + 1], dtype=torch.int32, device=device,
    )
    # Level 1 (tail): K groups, each 1 query attending L_tail tokens.
    qo_indptr_L1 = torch.arange(0, K + 1, dtype=torch.int32, device=device)
    kvp_indptr_L1 = torch.arange(
        0, (K + 1) * n_pages_tail_each, n_pages_tail_each,
        dtype=torch.int32, device=device,
    )
    kvi_L1 = (n_pages_prefix +
              torch.arange(K * n_pages_tail_each, dtype=torch.int32, device=device))
    kvl_L1 = torch.full((K,), ((L_tail - 1) % page_size) + 1,
                         dtype=torch.int32, device=device)

    wrapper2 = FusedMultiLevelCascadeAttentionWrapper(
        num_levels=2, float_workspace_buffer=workspace,
        kv_layout="NHD", device=device, max_levels=2,
    )
    wrapper2.plan(
        qo_indptr_arr=[qo_indptr_L0, qo_indptr_L1],
        paged_kv_indptr_arr=[kvp_indptr_L0, kvp_indptr_L1],
        paged_kv_indices_arr=[kvi_L0, kvi_L1],
        paged_kv_last_page_len=[kvl_L0, kvl_L1],
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, page_size=page_size, causal=False,
        q_data_type=dtype, kv_data_type=dtype,
        force_cta_tile_q=force_T,
    )
    q2 = torch.randn(K + K, num_qo_heads, head_dim, dtype=dtype, device=device)
    T_num2 = _time_us(lambda: wrapper2.run(q2, kv), device=device)

    # ---- num_levels=1 plan: L0 only ----
    wrapper1 = FusedMultiLevelCascadeAttentionWrapper(
        num_levels=1, float_workspace_buffer=workspace,
        kv_layout="NHD", device=device, max_levels=2,
    )
    wrapper1.plan(
        qo_indptr_arr=[qo_indptr_L0],
        paged_kv_indptr_arr=[kvp_indptr_L0],
        paged_kv_indices_arr=[kvi_L0],
        paged_kv_last_page_len=[kvl_L0],
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, page_size=page_size, causal=False,
        q_data_type=dtype, kv_data_type=dtype,
        force_cta_tile_q=force_T,
    )
    q1 = torch.randn(K, num_qo_heads, head_dim, dtype=dtype, device=device)
    T_num1 = _time_us(lambda: wrapper1.run(q1, kv), device=device)

    return {
        "K": K, "L_p": L_p, "L_tail": L_tail,
        "force_T": force_T,
        "force_l0_chunks": force_l0_chunks if force_l0_chunks else 0,
        "T_num2_us": T_num2,
        "T_num1_us": T_num1,
        "L1_marginal_us": T_num2 - T_num1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--sweep-l0-chunks",
        action="store_true",
        help="Sweep FLASHINFER_FORCE_KV_CHUNKS for L0 on a representative "
             "shape (K=32, L_p=32768, L_tail=128). Reveals whether the "
             "Config A vs Config B Δ collapses when L0's num_chunks is "
             "pinned identically in both wrappers.",
    )
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"     SMs: {torch.cuda.get_device_properties(device).multi_processor_count}")
    print()

    if args.sweep_l0_chunks:
        # Pin L0's num_chunks identically across Config A and Config B
        # for one representative shape. If Δ stays large → the split-K
        # planner is NOT the cause; if Δ collapses to ~L1's marginal
        # compute → it is.
        print("=== L0 split-K sweep on K=32, L_p=32768, L_tail=128 (force_T derived) ===\n")
        print(f"{'N_chunks_L0':>11} | {'T_num2 µs':>10} {'T_num1 µs':>10} | {'Δ µs':>8}")
        print("-" * 55)
        # 0 means "use the wrapper's default heuristic".
        for N in (0, 1, 2, 4, 8, 16, 32):
            r = probe(
                device, K=32, L_p=32768, L_tail=128,
                force_l0_chunks=N if N > 0 else None,
            )
            label = "default" if N == 0 else str(N)
            print(f"{label:>11} | "
                  f"{r['T_num2_us']:>10.2f} {r['T_num1_us']:>10.2f} | "
                  f"{r['L1_marginal_us']:>8.2f}")
        return

    # Shapes spanning C3-relevant regime.
    # C3 actual: K=32, L_p≈86000 (per rank), L_tail≈153 mid-decode.
    shapes = [
        # (K,  L_p,    L_tail)
        (16,  8192,   128),
        (32,  8192,   128),
        (32,  8192,   256),
        (32,  32768,  128),
        (32,  32768,  256),
        (64,  8192,   128),
        (64,  32768,  128),
        # Closer to C3 actual shape (skip if too big; FlashInfer workspace OOM):
        # (32, 86000, 153),
    ]

    print(f"{'K':>4} {'L_p':>8} {'L_tail':>7} {'T':>4} | {'T_num2 µs':>10} {'T_num1 µs':>10} | "
          f"{'L1 marginal µs':>14} | {'pred L1 µs':>10} | {'ratio':>8}")
    print("-" * 95)
    # Current cost model predicts L1 marginal cost ≈
    #   per_tile_per_kv_us[64] × cta_tile_kv × n_subtiles_for_L1 / num_sms_used
    # with K groups × 1 query per group × ⌈L_tail/64⌉ kv-chunks → K × ⌈L_tail/64⌉ sub-tiles
    SLOPE = 0.014349   # per_tile_per_kv_us[64]
    CTA_K = 64
    NUM_SMS = 132

    for (K, L_p, L_tail) in shapes:
        r = probe(device, K=K, L_p=L_p, L_tail=L_tail)
        n_subtiles = K * max(1, (L_tail + CTA_K - 1) // CTA_K)
        waves = max(1, (n_subtiles + NUM_SMS - 1) // NUM_SMS)
        tau_elem = SLOPE * CTA_K
        pred_L1_us = waves * tau_elem
        ratio = r["L1_marginal_us"] / pred_L1_us if pred_L1_us > 0 else float("nan")
        print(f"{K:>4} {L_p:>8} {L_tail:>7} {r['force_T']:>4} | "
              f"{r['T_num2_us']:>10.2f} {r['T_num1_us']:>10.2f} | "
              f"{r['L1_marginal_us']:>14.2f} | "
              f"{pred_L1_us:>10.2f} | "
              f"{ratio:>8.1f}x")


if __name__ == "__main__":
    main()

"""Sub-phase breakdown of MLCA plan_decode_step vs bs_kernel's fused 2L plan.

Constructs a synthetic but realistic decoding state (B prompts × K beams,
each beam holding L_p+t pages) and times each phase of:

  - MlcaBackend.plan_decode_step (non-fused 2L wrapper)
  - AdaptivePoolBackend.plan_decode_step (fused 2L wrapper, same input)

The two backends consume the same _pack_batched_cascade_arrays output, so
the only structural difference is the wrapper plan call: non-fused
iterates two BatchPrefill plans (L levels), fused calls a single
_plan_inner. Per-phase timers use torch.cuda.synchronize before reading
perf_counter.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import torch

from flashinfer import (
    FusedMultiLevelCascadeAttentionWrapper,
    MultiLevelCascadeAttentionWrapper,
)

from beam_engine.methods.adaptive_pool import (
    _adaptive_levels,
    _pack_batched_cascade_arrays,
)


def _make_workload(B: int, K: int, L_p_pages: int, page_size: int, *, device):
    """Build synthetic pages_prefix/pages_tail per beam.

    Mimics what page_driver hands the backend at step T: each prompt has
    a shared prefix of L_p_pages pages; each of K beams holds 1 unique
    tail page (the per-beam split-page CoW result).
    """
    pages_prefix_per_prompt = []
    pages_tail_per_prompt = []
    next_page = 0
    for _b in range(B):
        prefix = list(range(next_page, next_page + L_p_pages))
        next_page += L_p_pages
        tails: list[list[int]] = []
        for _k in range(K):
            tails.append([next_page])
            next_page += 1
        pages_prefix_per_prompt.append(prefix)
        pages_tail_per_prompt.append(tails)
    return pages_prefix_per_prompt, pages_tail_per_prompt, next_page


def _sync_now():
    torch.cuda.synchronize()
    return time.perf_counter()


def _time_fn(fn: Callable, iters: int):
    # warmup
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = _sync_now()
        fn()
        t1 = _sync_now()
        ts.append((t1 - t0) * 1000.0)
    ts.sort()
    return ts


def run(B: int, K: int, L_p_pages: int, iters: int,
        num_qo_heads: int, num_kv_heads: int, head_dim: int, page_size: int):
    device = torch.device("cuda")
    dtype = torch.float16

    pages_prefix_pp, pages_tail_pp, total_pages = _make_workload(
        B, K, L_p_pages, page_size, device=device,
    )

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    mlca_wrapper = MultiLevelCascadeAttentionWrapper(
        num_levels=2, float_workspace_buffer=workspace, kv_layout="NHD",
    )
    fused_wrapper = FusedMultiLevelCascadeAttentionWrapper(
        num_levels=2, float_workspace_buffer=workspace, kv_layout="NHD",
    )

    last_lca = [0] * B

    # ----- Phase 1: per-prompt _adaptive_levels loop (CPU). -----
    def phase_adaptive_levels():
        local_levels = []
        local_lca = list(last_lca)
        for b in range(B):
            lpl_per_beam = [page_size] * K
            levels, _order, lca_b = _adaptive_levels(
                pages_prefix_pp[b], pages_tail_pp[b], lpl_per_beam, K,
                max_levels=2, start_lca=local_lca[b],
            )
            local_lca[b] = lca_b
            local_levels.append(levels)
        return local_levels

    # Pre-compute levels once for downstream phases.
    levels_pp = phase_adaptive_levels()

    # ----- Phase 2: _pack_batched_cascade_arrays (host pack + H2D). -----
    def phase_pack():
        # _pack_batched_cascade_arrays now returns 6 entries (device tensors
        # + per-level host cumsum lists); discard the host lists here.
        out = _pack_batched_cascade_arrays(levels_pp, page_size, device)
        return out[:4]

    qo_arr, kvp_arr, kvi_arr, kvl_arr = phase_pack()

    # ----- Phase 3a: MLCA wrapper plan call. -----
    def phase_mlca_plan():
        mlca_wrapper.plan(
            qo_indptr_arr=qo_arr,
            paged_kv_indptr_arr=kvp_arr,
            paged_kv_indices_arr=kvi_arr,
            paged_kv_last_page_len=kvl_arr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            causal=False,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

    # ----- Phase 3b: Fused wrapper plan call (same input). -----
    def phase_fused_plan():
        fused_wrapper.plan(
            qo_indptr_arr=qo_arr,
            paged_kv_indptr_arr=kvp_arr,
            paged_kv_indices_arr=kvi_arr,
            paged_kv_last_page_len=kvl_arr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            causal=False,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

    # ----- Phase 4: write_pi/write_po build (per beam). -----
    def phase_write_metadata():
        write_pi = []
        write_po = []
        for b in range(B):
            for tails in pages_tail_pp[b]:
                write_pi.append(tails[-1])
                write_po.append(page_size - 1)
        torch.tensor(write_pi, dtype=torch.int32, device=device)
        torch.tensor(write_po, dtype=torch.int32, device=device)

    # ----- Combined: MLCA-style full plan_decode_step (single call). -----
    def combined_mlca():
        phase_adaptive_levels()
        out = phase_pack()
        qo, kp, ki, kl = out
        mlca_wrapper.plan(
            qo_indptr_arr=qo, paged_kv_indptr_arr=kp,
            paged_kv_indices_arr=ki, paged_kv_last_page_len=kl,
            num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, page_size=page_size, causal=False,
            q_data_type=dtype, kv_data_type=dtype,
        )
        phase_write_metadata()

    def combined_fused():
        phase_adaptive_levels()
        out = phase_pack()
        qo, kp, ki, kl = out
        fused_wrapper.plan(
            qo_indptr_arr=qo, paged_kv_indptr_arr=kp,
            paged_kv_indices_arr=ki, paged_kv_last_page_len=kl,
            num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, page_size=page_size, causal=False,
            q_data_type=dtype, kv_data_type=dtype,
        )
        phase_write_metadata()

    results = {}
    for name, fn in [
        ("phase1_adaptive_levels", phase_adaptive_levels),
        ("phase2_pack_arrays", phase_pack),
        ("phase3a_mlca_plan", phase_mlca_plan),
        ("phase3b_fused_plan", phase_fused_plan),
        ("phase4_write_metadata", phase_write_metadata),
        ("combined_mlca", combined_mlca),
        ("combined_fused", combined_fused),
    ]:
        ts = _time_fn(fn, iters)
        results[name] = (ts[len(ts) // 2], sum(ts) / len(ts), min(ts), max(ts))

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", default="1B-fewshot",
                   choices=["1B-fewshot", "1B-chain", "8B-fewshot",
                            "8B-chain", "1B-levelsys", "8B-levelsys"])
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    page_size = 16

    # Cells from paper-exp grid (K=64). L_p_pages = L_p / page_size.
    cells = {
        "1B-fewshot":   dict(B=32, K=64, L_p_pages=2048 // 16,
                             num_qo_heads=32, num_kv_heads=8, head_dim=64),
        "1B-chain":     dict(B=32, K=64, L_p_pages=2574 // 16,
                             num_qo_heads=32, num_kv_heads=8, head_dim=64),
        "1B-levelsys":  dict(B=4,  K=64, L_p_pages=2048 // 16,
                             num_qo_heads=32, num_kv_heads=8, head_dim=64),
        "8B-fewshot":   dict(B=8,  K=64, L_p_pages=2048 // 16,
                             num_qo_heads=32, num_kv_heads=8, head_dim=128),
        "8B-chain":     dict(B=8,  K=64, L_p_pages=2573 // 16,
                             num_qo_heads=32, num_kv_heads=8, head_dim=128),
        "8B-levelsys":  dict(B=4,  K=64, L_p_pages=2048 // 16,
                             num_qo_heads=32, num_kv_heads=8, head_dim=128),
    }
    cell = cells[args.cell]

    print(f"=== MLCA plan-phase breakdown: cell={args.cell} ===")
    print(f"    B={cell['B']}  K={cell['K']}  L_p_pages={cell['L_p_pages']}  "
          f"num_qo_heads={cell['num_qo_heads']}  num_kv_heads={cell['num_kv_heads']}  "
          f"head_dim={cell['head_dim']}  page_size={page_size}  iters={args.iters}")
    print()

    results = run(
        iters=args.iters, page_size=page_size,
        **cell,
    )

    rows = list(results.items())
    name_w = max(len(n) for n, _ in rows)
    print(f"  {'phase':<{name_w}}  {'p50 ms':>9}  {'mean ms':>9}  {'min ms':>9}  {'max ms':>9}")
    print(f"  {'-'*name_w}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
    for name, (p50, mean, lo, hi) in rows:
        print(f"  {name:<{name_w}}  {p50:9.4f}  {mean:9.4f}  {lo:9.4f}  {hi:9.4f}")

    # Derived: MLCA breakdown.
    p1 = results["phase1_adaptive_levels"][0]
    p2 = results["phase2_pack_arrays"][0]
    p3a = results["phase3a_mlca_plan"][0]
    p3b = results["phase3b_fused_plan"][0]
    p4 = results["phase4_write_metadata"][0]
    total_mlca = p1 + p2 + p3a + p4
    total_fused = p1 + p2 + p3b + p4
    print()
    print(f"  MLCA  per-step sum:   {total_mlca:.4f} ms  "
          f"(p1 {p1:.3f} + p2 {p2:.3f} + p3a {p3a:.3f} + p4 {p4:.3f})")
    print(f"  FUSED per-step sum:   {total_fused:.4f} ms  "
          f"(p1 {p1:.3f} + p2 {p2:.3f} + p3b {p3b:.3f} + p4 {p4:.3f})")
    if total_fused > 0:
        print(f"  Gap (mlca − fused):   {total_mlca - total_fused:.4f} ms  "
              f"(plan-call only: {p3a - p3b:.4f} ms, "
              f"{(p3a / p3b if p3b > 0 else float('inf')):.2f}× of fused)")
    print()
    print(f"  combined_mlca p50:    {results['combined_mlca'][0]:.4f} ms")
    print(f"  combined_fused p50:   {results['combined_fused'][0]:.4f} ms")


if __name__ == "__main__":
    main()

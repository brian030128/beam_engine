"""Probe the FlashInfer BatchDecodeWithPagedKVCacheWrapper directly to
isolate per-CTA serial setup (1) from effective HBM bandwidth at small R (2).

We sweep R ∈ {1, 16, 64, 132, 264, 528, 1056, 2112} with L_kv held at one of
several values. For each (R, L_kv) we measure per-call wall time.

Decomposition the wall time should satisfy:

  wall(R, L_kv) ≈ ν_decode_launch + ⌈R/N_SM⌉ · (κ_per_cta + L_kv · ρ_d_inner)

So:
  - **fix L_kv, vary R**: wall vs ⌈R/N_SM⌉ has slope = (κ + L_kv · ρ_d_inner)
    and intercept = ν_decode_launch. At small L_kv the slope ≈ κ alone, so a
    short-L_kv sweep isolates κ_per_cta (issue 1).
  - **effective bandwidth = (R · L_kv · bytes_per_kv) / wall_time**: at small
    R this should be <<peak ``B_hbm``; the ratio characterises issue 2.

We also include a back-to-back R=N_SM·k sweep so the "wave count" axis is
clean.
"""

from __future__ import annotations

import argparse
import csv
import sys

import torch
import flashinfer

DEFAULT_NUM_KV_HEADS = 8
DEFAULT_NUM_QO_HEADS = 32   # 8B GQA ratio 4
DEFAULT_HEAD_DIM = 128
DEFAULT_PAGE_SIZE = 16
NUM_SMS_DEFAULT = 132       # H100


def time_event_us(fn, n: int = 50, warmup: int = 5, device: torch.device | None = None) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000.0 / n  # ms → µs, amortised


def measure_decode(
    R: int, L_kv: int,
    *, num_kv_heads: int, num_qo_heads: int, head_dim: int,
    page_size: int, dtype: torch.dtype, device: torch.device, n_iters: int = 50,
) -> float:
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_tensor_cores=True,
    )
    num_pages_per_req = max(1, (L_kv + page_size - 1) // page_size)
    total_pages = R * num_pages_per_req
    kv = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    indptr = torch.arange(
        0, (R + 1) * num_pages_per_req, num_pages_per_req,
        dtype=torch.int32, device=device,
    )
    indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    last_page_len = torch.full(
        (R,),
        max(1, L_kv - (num_pages_per_req - 1) * page_size),
        dtype=torch.int32, device=device,
    )
    wrapper.plan(
        indptr=indptr, indices=indices, last_page_len=last_page_len,
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, page_size=page_size,
    )
    q = torch.randn(R, num_qo_heads, head_dim, dtype=dtype, device=device)
    us = time_event_us(lambda: wrapper.run(q, kv), n=n_iters, device=device)
    return us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/bs_kernel/results/paper-exp/probe_decode_kernel.csv")
    ap.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    ap.add_argument("--num-qo-heads", type=int, default=DEFAULT_NUM_QO_HEADS)
    ap.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    ap.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    bytes_per_kv = 2 * args.num_kv_heads * args.head_dim * 2  # K+V × heads × dim × fp16
    print(f"GPU: {torch.cuda.get_device_name(device)}  num_sms={num_sms}")
    print(f"bytes_per_kv_token = {bytes_per_kv} (num_kv_heads={args.num_kv_heads}, head_dim={args.head_dim}, fp16)")
    print()

    # R grid spans sub-wave (R<N_SM), exactly-N_SM, and multi-wave regimes.
    R_grid = [1, 4, 16, 32, 64, num_sms, 2*num_sms, 4*num_sms, 8*num_sms, 16*num_sms]
    L_kv_grid = [16, 64, 256, 1024, 4096, 16384]

    rows: list[dict] = []
    for L_kv in L_kv_grid:
        for R in R_grid:
            try:
                us = measure_decode(
                    R, L_kv,
                    num_kv_heads=args.num_kv_heads, num_qo_heads=args.num_qo_heads,
                    head_dim=args.head_dim, page_size=args.page_size,
                    dtype=dtype, device=device, n_iters=args.iters,
                )
            except Exception as e:
                print(f"  R={R:>6d} L_kv={L_kv:>6d}  FAILED: {type(e).__name__}: {e}")
                continue
            waves = -(-R // num_sms)   # ceil
            bytes_loaded = R * L_kv * bytes_per_kv
            eff_bw_bps = bytes_loaded / (us * 1e-6)
            eff_bw_pct_peak = eff_bw_bps / 1.3e12 * 100   # vs ~1.3 TB/s HBM3
            rows.append({
                "R": R, "L_kv": L_kv, "waves": waves,
                "wall_us": round(us, 3),
                "eff_bw_gb_per_s": round(eff_bw_bps / 1e9, 1),
                "eff_bw_pct_peak": round(eff_bw_pct_peak, 1),
                "bytes_loaded": bytes_loaded,
            })
            print(
                f"  R={R:>6d}  L_kv={L_kv:>6d}  waves={waves:>3d}  "
                f"wall={us:>9.2f} µs   "
                f"eff_BW={eff_bw_bps / 1e9:>7.1f} GB/s  ({eff_bw_pct_peak:>5.1f}% of peak)"
            )
            torch.cuda.empty_cache()
        print()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

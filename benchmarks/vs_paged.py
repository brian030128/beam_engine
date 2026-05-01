"""Per-decode-step micro-benchmark: fused-cascade vs. standard paged attention.

For each (K beams, L_p prompt, tail) configuration we measure the time to
run ONE decode step of attention. Three contenders:

  1. **Paged attention** — BatchDecodeWithPagedKVCacheWrapper. Each beam is
     an independent sequence; the shared prompt pages are referenced by
     every beam's `kv_indices` (page sharing via refcount, what
     baselines/paged.py uses). Memory traffic per step:
       O(K · L_p)   prompt KV (each beam re-reads it)
     + O(K · tail)  per-beam tail KV.

  2. **Cascade (baseline)** — MultiLevelCascadeAttentionWrapper. Explicit
     2-level cascade: prompt at L0 (one shared scan), tails at L1 (per
     beam). Memory traffic:
       O(L_p)        prompt KV (shared scan, read once)
     + O(K · tail)   per-beam tails.

  3. **Cascade (fused, our work)** — FusedMultiLevelCascadeAttentionWrapper.
     Same algorithmic structure as (2) but collapses the per-level
     prefill launches into one fused kernel + a custom persistent
     merge. Same memory traffic as (2); fewer kernel launches.

Sweeps cover the typical post-prefill / mid-decode regime: K∈{4,8,16,32,64},
L_p∈{1024,4096,8192}, tail∈{1,64,256}.

Per CLAUDE.md: pick a fully idle GPU and pin via CUDA_VISIBLE_DEVICES.

Usage:
    nvidia-smi
    PYTHONPATH=3rdparty:$PYTHONPATH CUDA_VISIBLE_DEVICES=<id> \\
        uv run python benchmarks/vs_paged.py
"""

from __future__ import annotations

import statistics

import torch

import flashinfer
from flashinfer.testing import bench_gpu_time


NUM_QO_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 16
DTYPE = torch.float16


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def make_workload(K: int, L_p: int, tail: int, device):
    """Build one workload usable by all three contenders.

    Returns:
      kv_cache: [n_pages, 2, page_size, num_kv_heads, head_dim] paged buffer.
      shared_pages: page indices [0, n_prompt) holding the shared prompt.
      per_beam_tail_pages: list of K lists, each containing that beam's
                           tail page indices.
      L_p_actual: prompt length (padded to a page boundary internally).
      tail_actual: per-beam tail length.
    """
    n_prompt = _ceil_div(L_p, PAGE_SIZE)
    n_tail = max(_ceil_div(tail, PAGE_SIZE), 1)
    total_pages = n_prompt + K * n_tail
    kv_cache = torch.randn(
        total_pages, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device=device,
    )
    shared_pages = list(range(n_prompt))
    per_beam_tail_pages = [
        list(range(n_prompt + i * n_tail, n_prompt + (i + 1) * n_tail))
        for i in range(K)
    ]
    return kv_cache, shared_pages, per_beam_tail_pages, n_prompt, n_tail


# ---------------------------------------------------------------------------
# Three planners
# ---------------------------------------------------------------------------


def plan_paged(K, shared_pages, per_beam_tail_pages, L_p, tail, device):
    """Standard paged decode: each beam = its own sequence with full
    [shared_prompt..., unique_tail...] page list."""
    indices = []
    indptr = [0]
    last_page_lens = []
    for i in range(K):
        beam_pages = shared_pages + per_beam_tail_pages[i]
        indices.extend(beam_pages)
        indptr.append(indptr[-1] + len(beam_pages))
        # Last page length = how many tokens are in the LAST page.
        # Each beam has L_p (in shared) + tail (in tail_pages) tokens total.
        # The last page is the last tail page, with `tail % page_size` valid
        # tokens (or page_size if tail is page-aligned and > 0).
        if tail > 0:
            last = ((tail - 1) % PAGE_SIZE) + 1
        else:
            # No tail: last page is the last prompt page.
            last = ((L_p - 1) % PAGE_SIZE) + 1 if L_p > 0 else PAGE_SIZE
        last_page_lens.append(last)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_tensor_cores=True
    )
    wrapper.plan(
        indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        indices=torch.tensor(indices, dtype=torch.int32, device=device),
        last_page_len=torch.tensor(last_page_lens, dtype=torch.int32, device=device),
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
    )
    return wrapper


def _build_cascade_inputs(K, shared_pages, per_beam_tail_pages, L_p, tail, device):
    """Common 2-level cascade inputs (used by both cascade variants).
    Level 0: 1 group of K beams, shared prompt.
    Level 1: K groups of 1 beam, per-beam tail.
    """
    n_prompt = len(shared_pages)
    n_tail = len(per_beam_tail_pages[0]) if per_beam_tail_pages and per_beam_tail_pages[0] else 1

    qo_arr = [
        torch.tensor([0, K], dtype=torch.int32, device=device),
        torch.arange(K + 1, dtype=torch.int32, device=device),
    ]
    # L0: 1 request, prompt pages [0, n_prompt).
    kvi_l0 = torch.tensor([0, n_prompt], dtype=torch.int32, device=device)
    kvix_l0 = torch.tensor(shared_pages, dtype=torch.int32, device=device)
    last_l0 = torch.tensor(
        [((L_p - 1) % PAGE_SIZE) + 1 if L_p > 0 else PAGE_SIZE],
        dtype=torch.int32, device=device,
    )
    # L1: K requests, each with its own tail pages.
    kvi_l1 = torch.arange(K + 1, dtype=torch.int32, device=device) * n_tail
    flat_tail = [p for beam in per_beam_tail_pages for p in beam]
    kvix_l1 = torch.tensor(flat_tail, dtype=torch.int32, device=device)
    last_l1 = torch.full(
        (K,), ((tail - 1) % PAGE_SIZE) + 1 if tail > 0 else 1,
        dtype=torch.int32, device=device,
    )
    return [qo_arr[0], qo_arr[1]], [kvi_l0, kvi_l1], [kvix_l0, kvix_l1], [last_l0, last_l1]


def plan_cascade_baseline(K, shared_pages, per_beam_tail_pages, L_p, tail, device):
    qo, kvi, kvix, last = _build_cascade_inputs(
        K, shared_pages, per_beam_tail_pages, L_p, tail, device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(2, workspace, "NHD")
    wrapper.plan(qo, kvi, kvix, last,
                 NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE)
    return wrapper


def plan_cascade_fused(K, shared_pages, per_beam_tail_pages, L_p, tail, device):
    qo, kvi, kvix, last = _build_cascade_inputs(
        K, shared_pages, per_beam_tail_pages, L_p, tail, device
    )
    wrapper = flashinfer.FusedMultiLevelCascadeAttentionWrapper(2, kv_layout="NHD")
    wrapper.plan(qo, kvi, kvix, last,
                 NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE)
    return wrapper


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


def _median_us(fn) -> float:
    times = bench_gpu_time(
        fn, dry_run_iters=5, repeat_iters=50, cold_l2_cache=False,
    )
    return statistics.median(times) * 1000.0


def run_one(K, L_p, tail, device):
    kv_cache, shared, per_beam, _, _ = make_workload(K, L_p, tail, device)
    q = torch.randn(K, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device)

    # Plan all three.
    paged = plan_paged(K, shared, per_beam, L_p, tail, device)
    casc_base = plan_cascade_baseline(K, shared, per_beam, L_p, tail, device)
    casc_fused = plan_cascade_fused(K, shared, per_beam, L_p, tail, device)

    t_paged = _median_us(lambda: paged.run(q, kv_cache))
    t_casc_base = _median_us(lambda: casc_base.run(q, kv_cache))
    t_casc_fused = _median_us(lambda: casc_fused.run(q, kv_cache))
    return t_paged, t_casc_base, t_casc_fused


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
    print()
    print("=" * 96)
    print("Per-decode-step latency: paged vs cascade-baseline vs cascade-fused")
    print("Workload: K beams, shared L_p prompt, tail tokens unique per beam.")
    print("=" * 96)
    header = (
        f"  {'K':>3} {'L_p':>5} {'tail':>5} | "
        f"{'paged':>9} {'cascade':>9} {'fused':>9} | "
        f"{'fused/paged':>11} {'fused/casc':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    Ks = [4, 8, 16, 32, 64]
    L_ps = [1024, 4096, 8192]
    tails = [1, 64, 256]

    speedups_vs_paged = []
    speedups_vs_casc = []
    for K in Ks:
        for L_p in L_ps:
            for tail in tails:
                try:
                    t_p, t_cb, t_cf = run_one(K, L_p, tail, device)
                except Exception as e:
                    print(f"  K={K} L_p={L_p} tail={tail}: FAIL {e}")
                    continue
                sp_paged = t_p / t_cf
                sp_casc = t_cb / t_cf
                speedups_vs_paged.append(sp_paged)
                speedups_vs_casc.append(sp_casc)
                print(
                    f"  {K:>3} {L_p:>5} {tail:>5} | "
                    f"{fmt(t_p):>9} {fmt(t_cb):>9} {fmt(t_cf):>9} | "
                    f"{sp_paged:>10.2f}x {sp_casc:>9.2f}x"
                )

    print()
    print("=" * 96)
    if speedups_vs_paged:
        print(f"Fused cascade vs paged attention:")
        print(
            f"  speedup   median={statistics.median(speedups_vs_paged):.2f}x  "
            f"mean={statistics.mean(speedups_vs_paged):.2f}x  "
            f"min={min(speedups_vs_paged):.2f}x  "
            f"max={max(speedups_vs_paged):.2f}x"
        )
        n_faster_paged = sum(1 for s in speedups_vs_paged if s > 1.0)
        print(f"  {n_faster_paged}/{len(speedups_vs_paged)} configs strictly faster than paged.")
    if speedups_vs_casc:
        print(f"Fused cascade vs MultiLevelCascade baseline:")
        print(
            f"  speedup   median={statistics.median(speedups_vs_casc):.2f}x  "
            f"mean={statistics.mean(speedups_vs_casc):.2f}x  "
            f"min={min(speedups_vs_casc):.2f}x  "
            f"max={max(speedups_vs_casc):.2f}x"
        )
        n_faster_casc = sum(1 for s in speedups_vs_casc if s > 1.0)
        print(f"  {n_faster_casc}/{len(speedups_vs_casc)} configs strictly faster than cascade baseline.")


if __name__ == "__main__":
    main()

"""Per-device cost-model coefficient calibration.

Microbenchmark probes for the device coefficients the cost model uses
(see :class:`cost_model.Coefficients`). Run once at engine init; cached
to ``~/.cache/beam_engine/coeffs-<gpu>.json`` and reused on subsequent
calls if the GPU model matches.

Probes (per the design plan):

* ``B_hbm``        HBM bandwidth via ``cudaMemcpyDtoD`` on a contiguous
                   slab.
* ``launch_us``    Empty Triton-kernel launch overhead (median over N
                   runs).
* ``sync_us``      ``torch.cuda.synchronize()`` call cost on an idle
                   stream.
* ``merge_us``     ``flashinfer.merge_state_in_place`` at a representative
                   shape (K rows × num_heads × head_dim).
* ``per_tile_us``  For T ∈ {T_SMALL, *T_LARGE_CHOICES}: time
                   ``BatchPrefillWithPagedKVCacheWrapper`` with K=T
                   queries against ``L_kv=1024`` tokens of K/V — treating
                   that as one T-row Q tile of work. The unmodified
                   FlashInfer prefill kernel auto-selects its internal
                   tile size; this probe measures *per-call cost at
                   varying Q-row counts*, which is the relative scaling
                   the cost model needs. Phase 2 (kernel mods) will let
                   us measure the modified kernel directly at each
                   compile-time T.

The kernel itself is **not** re-tuned per device — FlashInfer's JIT
already picks register count, shared-memory layout, and instruction
selection per SM. This calibration only measures behaviors the JIT
cannot specialize for cost-model-time decisions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .cost_model import Coefficients, T_LARGE_CHOICES, T_SMALL


_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "beam_engine"


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _cache_path(device: torch.device) -> Path:
    name = _sanitize(torch.cuda.get_device_name(device))
    return _DEFAULT_CACHE_DIR / f"coeffs-{name}.json"


def _to_payload(c: Coefficients) -> dict:
    return {
        "B_hbm": c.B_hbm,
        "launch_us": c.launch_us,
        "sync_us": c.sync_us,
        "merge_us": c.merge_us,
        "per_tile_us": {str(k): v for k, v in c.per_tile_us.items()},
        "per_beam_us_per_kv_token": c.per_beam_us_per_kv_token,
        "num_sms": c.num_sms,
        "share_extra_us": c.share_extra_us,
        "dual_pool_extra_us": c.dual_pool_extra_us,
        "dec_tail_extra_us": c.dec_tail_extra_us,
        "dec_tail_per_tail_page_us": c.dec_tail_per_tail_page_us,
        "decode_us_per_beam_kv_token": c.decode_us_per_beam_kv_token,
        "decode_launch_us": c.decode_launch_us,
        "bw_efficiency_floor": c.bw_efficiency_floor,
        "max_dispatch_depth": c.max_dispatch_depth,
    }


def _from_payload(d: dict) -> Coefficients:
    return Coefficients(
        B_hbm=d["B_hbm"],
        launch_us=d["launch_us"],
        sync_us=d["sync_us"],
        merge_us=d["merge_us"],
        per_tile_us={int(k): v for k, v in d["per_tile_us"].items()},
        per_beam_us_per_kv_token=d.get("per_beam_us_per_kv_token", 0.0008),
        num_sms=d.get("num_sms", 84),
        share_extra_us=d.get("share_extra_us", 0.0),
        dual_pool_extra_us=d.get("dual_pool_extra_us", 0.0),
        dec_tail_extra_us=d.get("dec_tail_extra_us", 0.0),
        dec_tail_per_tail_page_us=d.get("dec_tail_per_tail_page_us", 0.0),
        decode_us_per_beam_kv_token=d.get("decode_us_per_beam_kv_token", 0.0008),
        decode_launch_us=d.get("decode_launch_us", 6.0),
        # H100 fit from bench_dispatch_grid measurements: floor=0.5 gives
        # 81% picker-vs-measurement match across K∈{16,64} × L_p∈{2K,8K,32K}
        # × B∈{1,8,32}. Cached coefficients files written before this
        # field existed default to the empirically-fit value.
        bw_efficiency_floor=d.get("bw_efficiency_floor", 0.5),
        # Default 3 reproduces legacy enumeration. Set higher (typically
        # 6) once the bs_kernel driver supports deeper cascades AND
        # workloads exercise hierarchical sharing depth>3.
        max_dispatch_depth=d.get("max_dispatch_depth", 3),
    )


def load_or_defaults(device: torch.device | str = "cuda") -> Coefficients:
    """Load device-specific coefficients from the on-disk cache; fall
    back to ``Coefficients.defaults()`` if no cache file exists for
    this GPU. The driver uses this when ``coefficients=None`` is
    passed to ``beam_search``.
    """
    device = torch.device(device)
    path = _cache_path(device)
    if path.exists():
        return _load(path)
    return Coefficients.defaults()


def _save(c: Coefficients, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_payload(c), indent=2))


def _load(path: Path) -> Coefficients:
    return _from_payload(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _time_event_us(fn, n: int, *, warmup: int = 5, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    end.synchronize()
    # elapsed_time returns ms; convert to µs and amortize.
    return start.elapsed_time(end) * 1000.0 / n


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def measure_B_hbm(
    device: torch.device,
    *,
    n_bytes: int = 256 * 1024 * 1024,
    n_runs: int = 5,
) -> float:
    src = torch.empty(n_bytes, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    us = _time_event_us(lambda: dst.copy_(src), n_runs, device=device)
    return n_bytes / us  # bytes / µs


def measure_launch_us(device: torch.device, *, n: int = 1000) -> float:
    import triton

    @triton.jit
    def _empty():
        pass

    return _time_event_us(lambda: _empty[(1,)](), n, device=device)


def measure_sync_us(device: torch.device, *, n: int = 1000) -> float:
    """Wall-clock cost of a torch.cuda.synchronize() on an idle stream.

    Uses time.perf_counter rather than cuda events because we're measuring
    the call itself, not GPU-side work.
    """
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n):
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6 / n


def measure_merge_us(
    device: torch.device,
    *,
    K: int = 8,
    num_heads: int = 32,
    head_dim: int = 128,
    n: int = 50,
    dtype: torch.dtype = torch.float16,
) -> float:
    from flashinfer import merge_state_in_place

    v_a = torch.randn(K, num_heads, head_dim, dtype=dtype, device=device)
    s_a = torch.randn(K, num_heads, dtype=torch.float32, device=device)
    v_b = torch.randn(K, num_heads, head_dim, dtype=dtype, device=device)
    s_b = torch.randn(K, num_heads, dtype=torch.float32, device=device)
    return _time_event_us(
        lambda: merge_state_in_place(v_a, s_a, v_b, s_b),
        n,
        device=device,
    )


def measure_decode_per_kv_us(
    device: torch.device,
    *,
    num_kv_heads: int = 8,
    num_qo_heads: int = 32,
    head_dim: int = 128,
    page_size: int = 16,
    n: int = 50,
    dtype: torch.dtype = torch.float16,
    BS_pairs: tuple[tuple[int, int], ...] = (
        (256, 256), (256, 1024), (2048, 64), (2048, 256),
    ),
) -> tuple[float, float]:
    """Linear regression of ``BatchDecodeWithPagedKVCacheWrapper`` time vs
    (BS × L_kv).

    Returns ``(launch_us, us_per_kv_token)``: the intercept and slope of
    the best-fit line ``time_us ≈ launch_us + us_per_kv_token *
    (BS × L_kv)``. Used by the cost model's DEC_TAIL pricing — at
    decode time CTA_Q=1 so per-tile padding is irrelevant; cost is
    bandwidth (proportional to KV bytes loaded) plus a per-call
    constant.
    """
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_tensor_cores=True,
    )
    xs: list[float] = []
    ys: list[float] = []
    for BS, L_kv in BS_pairs:
        num_pages_per_req = max(1, L_kv // page_size)
        # Each request reads `num_pages_per_req` pages, all distinct
        # (paged-KV needs unique page indices per request, but kernel
        # cost depends only on per-request KV length).
        total_pages = BS * num_pages_per_req
        kv = torch.randn(
            total_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )
        indptr = torch.arange(
            0, (BS + 1) * num_pages_per_req, num_pages_per_req,
            dtype=torch.int32, device=device,
        )
        indices = torch.arange(total_pages, dtype=torch.int32, device=device)
        last_page_len = torch.full(
            (BS,), page_size, dtype=torch.int32, device=device,
        )
        wrapper.plan(
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
        )
        q = torch.randn(BS, num_qo_heads, head_dim, dtype=dtype, device=device)
        us = _time_event_us(lambda: wrapper.run(q, kv), n, device=device)
        xs.append(float(BS * L_kv))
        ys.append(us)
    # Closed-form least squares: minimise sum (y - (a + b x))^2.
    n_pts = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n_pts * sxx - sx * sx
    if denom == 0:
        slope = ys[0] / xs[0] if xs[0] > 0 else 0.0
        intercept = 0.0
    else:
        slope = (n_pts * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n_pts
    return max(0.0, intercept), max(0.0, slope)


def measure_per_tile_us(
    device: torch.device,
    T: int,
    *,
    num_kv_heads: int = 8,
    num_qo_heads: int = 32,
    head_dim: int = 128,
    L_kv: int = 1024,
    page_size: int = 16,
    n: int = 50,
    dtype: torch.dtype = torch.float16,
) -> float:
    """Approximate per-tile cost for a T-row Q tile.

    Runs ``BatchPrefillWithPagedKVCacheWrapper`` with K=T queries × L_kv
    KV tokens. The unmodified prefill kernel auto-selects its internal
    tile size; this probe gives a *relative* scaling of "attention call
    cost as a function of T" that the cost model can use for its
    pool-count and T-large picks. Phase 2 will re-measure against the
    modified fused-cascade kernel where we control T directly.
    """
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    num_pages = L_kv // page_size
    kv = torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr=torch.tensor([0, T], dtype=torch.int32, device=device),
        paged_kv_indptr=torch.tensor([0, num_pages], dtype=torch.int32, device=device),
        paged_kv_indices=torch.arange(num_pages, dtype=torch.int32, device=device),
        paged_kv_last_page_len=torch.tensor([page_size], dtype=torch.int32, device=device),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=False,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    q = torch.randn(T, num_qo_heads, head_dim, dtype=dtype, device=device)
    return _time_event_us(lambda: wrapper.run(q, kv), n, device=device)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def calibrate(
    device: str | torch.device = "cuda",
    *,
    force: bool = False,
    cache_path: Path | None = None,
    verbose: bool = False,
) -> Coefficients:
    """Run all probes and cache the result.

    Reuses the on-disk cache if present (keyed by GPU model name) unless
    ``force=True``.
    """
    device = torch.device(device)
    path = cache_path or _cache_path(device)
    if not force and path.exists():
        if verbose:
            print(f"[calibrate] reusing cache at {path}")
        return _load(path)

    if verbose:
        print(f"[calibrate] probing {torch.cuda.get_device_name(device)}")

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    decode_launch, decode_per_kv = measure_decode_per_kv_us(device)
    coeffs = Coefficients(
        B_hbm=measure_B_hbm(device),
        launch_us=measure_launch_us(device),
        sync_us=measure_sync_us(device),
        merge_us=measure_merge_us(device),
        per_tile_us={
            T: measure_per_tile_us(device, T)
            for T in (T_SMALL, *T_LARGE_CHOICES)
        },
        num_sms=num_sms,
        decode_us_per_beam_kv_token=decode_per_kv,
        decode_launch_us=decode_launch,
    )
    _save(coeffs, path)
    if verbose:
        print(f"[calibrate] wrote {path}")
    return coeffs


def _print_coeffs(c: Coefficients) -> None:
    print(f"  B_hbm        = {c.B_hbm:>10.1f} bytes/µs   ({c.B_hbm / 1e6:.2f} TB/s)")
    print(f"  launch_us    = {c.launch_us:>10.2f}")
    print(f"  sync_us      = {c.sync_us:>10.2f}")
    print(f"  merge_us     = {c.merge_us:>10.2f}")
    for T, t in sorted(c.per_tile_us.items()):
        print(f"  per_tile[{T:3d}] = {t:>10.2f} µs")
    print(f"  decode_launch_us         = {c.decode_launch_us:>10.2f}")
    print(f"  decode_us_per_beam_kv    = {c.decode_us_per_beam_kv_token:>10.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Calibrate bs_kernel cost-model coefficients.")
    ap.add_argument("--force", action="store_true", help="re-run probes and overwrite cache")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    c = calibrate(args.device, force=args.force, verbose=True)
    _print_coeffs(c)

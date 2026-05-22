"""Per-device cost-model coefficient calibration.

Microbenchmark probes for the device coefficients the cost model uses
(see :class:`cost_model.Coefficients`). Run once at engine init; cached
to ``~/.cache/beam_engine/coeffs-<gpu>.json`` and reused on subsequent
calls if the GPU model matches.

Probes (per the design plan):

* ``B_hbm``        HBM bandwidth via ``cudaMemcpyDtoD`` on a contiguous
                   slab.
* ``merge_launch_us``, ``merge_bw_us_per_row``
                   ``flashinfer.merge_state_in_place`` cost decomposed
                   into per-launch intercept and per-row slope (linear
                   fit over a small K-sweep).
* ``decode_launch_us``, ``decode_us_per_beam_kv_token``
                   ``BatchDecodeWithPagedKVCacheWrapper`` cost decomposed
                   into per-launch intercept and per-(beam,kv-token)
                   slope (linear fit over an L_kv × batch grid).
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
from pathlib import Path

import torch

from .cost_model import (
    Coefficients, T_LARGE_CHOICES, T_SMALL,
    PER_TILE_L_KV_REF, PER_TILE_L_KV_LONG,
    DTYPE_FP16, DTYPE_BF16, DTYPE_FP8_E4M3,
    DEFAULT_KV_DTYPE, canonical_kv_dtype,
)


# Dtypes the calibration sweeps by default. Each entry is the canonical
# label paired with the torch.dtype to probe. fp8 is included so 70B-fp8
# cells get their own per_tile and decode rates — without it the picker
# falls back to fp16 rates and mispicks (see project memory
# [[project_70b_fp8_picker_recalibrate]]). If FlashInfer rejects fp8 in
# the probe path on a given build, the dtype is skipped with a warning.
def _default_probe_dtypes() -> list[tuple[str, "torch.dtype"]]:
    probes: list[tuple[str, "torch.dtype"]] = [
        (DTYPE_FP16, torch.float16),
        (DTYPE_BF16, torch.bfloat16),
    ]
    if hasattr(torch, "float8_e4m3fn"):
        probes.append((DTYPE_FP8_E4M3, torch.float8_e4m3fn))
    return probes


_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "beam_engine"


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _cache_path(
    device: torch.device,
    model: str | None = None,
    tp_size: int | None = None,
) -> Path:
    """Cache filename. Keyed on GPU × model × TP topology, since each
    factor independently shifts the autotuned overheads:

    - GPU: launch / sync / HBM-bandwidth constants are device-specific.
    - model: num_layers and head_dim shift the FUSED/DEC_TAIL crossover
      via per-step launch count and per-tile arithmetic intensity.
    - tp_size: under TP the per-rank ``num_kv_heads`` and
      ``num_qo_heads`` get divided, which changes the per-tile cost
      and the cascade-merge cost the picker sees.

    Falls back to shorter keys when args are omitted so pre-existing
    caches keep working:

    - ``coeffs-<gpu>.json``               (legacy, GPU only)
    - ``coeffs-<gpu>-<model>.json``       (GPU + model, tp=1 implicit)
    - ``coeffs-<gpu>-<model>-tp<N>.json`` (full key, N >= 2)
    """
    gpu = _sanitize(torch.cuda.get_device_name(device))
    if model is None:
        return _DEFAULT_CACHE_DIR / f"coeffs-{gpu}.json"
    base = f"coeffs-{gpu}-{_sanitize(model)}"
    if tp_size is not None and tp_size > 1:
        base = f"{base}-tp{tp_size}"
    return _DEFAULT_CACHE_DIR / f"{base}.json"


def _to_payload(c: Coefficients) -> dict:
    return {
        "B_hbm": c.B_hbm,
        "merge_launch_us": c.merge_launch_us,
        "merge_bw_us_per_row": c.merge_bw_us_per_row,
        # Per-dtype nested maps. Outer key = canonical dtype label
        # (fp16/bf16/fp8_e4m3/...); inner key = tile dim T as str (JSON
        # has no int keys). Older caches stored these as flat ``{T_str:
        # rate}`` maps; ``_from_payload`` accepts both shapes.
        "per_tile_us": {
            dt: {str(t): v for t, v in inner.items()}
            for dt, inner in c.per_tile_us.items()
        },
        "per_tile_per_kv_us": {
            dt: {str(t): v for t, v in inner.items()}
            for dt, inner in c.per_tile_per_kv_us.items()
        },
        "num_sms": c.num_sms,
        # Per-dtype scalar map.
        "decode_us_per_beam_kv_token": dict(c.decode_us_per_beam_kv_token),
        "decode_launch_us": c.decode_launch_us,
        "max_dispatch_depth": c.max_dispatch_depth,
    }


def _is_dtype_nested(payload_value) -> bool:
    """A nested dtype map looks like ``{"fp16": {"64": 1.4, ...}}``; a
    flat (legacy) map looks like ``{"64": 1.4, ...}``. Distinguish by
    checking whether the values are themselves dicts.
    """
    if not isinstance(payload_value, dict) or not payload_value:
        return False
    return any(isinstance(v, dict) for v in payload_value.values())


def _parse_per_tile_table(payload_value) -> dict:
    """Parse either a nested (per-dtype) or flat (legacy) per_tile table.

    Legacy flat tables ``{T_str: rate}`` are promoted to
    ``{DEFAULT_KV_DTYPE: {T_int: rate}}`` so the in-memory format is
    uniform.
    """
    if payload_value is None:
        return {}
    if _is_dtype_nested(payload_value):
        return {
            dt: {int(t): float(v) for t, v in inner.items()}
            for dt, inner in payload_value.items()
        }
    # Legacy flat shape: treat as the default dtype.
    return {DEFAULT_KV_DTYPE: {int(t): float(v) for t, v in payload_value.items()}}


def _parse_decode_per_token(payload_value) -> dict:
    """Parse either a nested per-dtype map or a legacy flat scalar."""
    if payload_value is None:
        return {DEFAULT_KV_DTYPE: 0.0008}
    if isinstance(payload_value, dict):
        return {dt: float(v) for dt, v in payload_value.items()}
    # Legacy scalar.
    return {DEFAULT_KV_DTYPE: float(payload_value)}


def _from_payload(d: dict) -> Coefficients:
    # Backward compat: older cache files stored ``merge_us`` as the only
    # merge cost (interpreted as launch overhead, bandwidth assumed 0).
    # Newer files store ``merge_launch_us`` + ``merge_bw_us_per_row``
    # explicitly; fall back to the legacy ``merge_us`` key if needed.
    legacy_merge_us = d.get("merge_us", 2.0)
    per_tile_us = _parse_per_tile_table(d.get("per_tile_us"))
    per_tile_per_kv_us = (
        _parse_per_tile_table(d["per_tile_per_kv_us"])
        if "per_tile_per_kv_us" in d
        # CTA-granularity wave model requires per_tile_per_kv_us. Older
        # caches omitted it (the legacy elemental-sub-tile model fell back
        # to per_tile_us / PER_TILE_L_KV_REF). Default 0 so cost_model
        # uses its fallback for legacy caches.
        else {DEFAULT_KV_DTYPE: {16: 0.0, 64: 0.0, 128: 0.0}}
    )
    decode_per_token = _parse_decode_per_token(
        d.get("decode_us_per_beam_kv_token")
    )
    return Coefficients(
        B_hbm=d["B_hbm"],
        merge_launch_us=d.get("merge_launch_us", legacy_merge_us),
        merge_bw_us_per_row=d.get("merge_bw_us_per_row", 0.0),
        per_tile_us=per_tile_us,
        per_tile_per_kv_us=per_tile_per_kv_us,
        num_sms=d.get("num_sms", 84),
        decode_us_per_beam_kv_token=decode_per_token,
        decode_launch_us=d.get("decode_launch_us", 6.0),
        # Default 3 reproduces legacy enumeration. Set higher (typically
        # 6) once the bs_kernel driver supports deeper cascades AND
        # workloads exercise hierarchical sharing depth>3.
        max_dispatch_depth=d.get("max_dispatch_depth", 3),
    )


def load_or_defaults(
    device: torch.device | str = "cuda",
    model: str | None = None,
    tp_size: int | None = None,
) -> Coefficients:
    """Load coefficients from the on-disk cache.

    Resolution order (most-specific → least-specific → defaults):
      1. ``coeffs-<gpu>-<model>-tp<N>.json``   (full key, N >= 2)
      2. ``coeffs-<gpu>-<model>.json``         (model-specific, TP=1)
      3. ``coeffs-<gpu>.json``                 (legacy GPU-only)
      4. ``Coefficients.defaults()``

    The driver uses this when ``coefficients=None`` is passed to
    ``beam_search``.
    """
    device = torch.device(device)
    if model is not None and tp_size is not None and tp_size > 1:
        tp_specific = _cache_path(device, model, tp_size)
        if tp_specific.exists():
            return _load(tp_specific)
    if model is not None:
        model_specific = _cache_path(device, model)
        if model_specific.exists():
            return _load(model_specific)
    legacy = _cache_path(device)
    if legacy.exists():
        return _load(legacy)
    return Coefficients.defaults()


def _save(c: Coefficients, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_payload(c), indent=2))


def _load(path: Path) -> Coefficients:
    return _from_payload(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _is_fp8(dtype) -> bool:
    fp8s = []
    if hasattr(torch, "float8_e4m3fn"):
        fp8s.append(torch.float8_e4m3fn)
    if hasattr(torch, "float8_e5m2"):
        fp8s.append(torch.float8_e5m2)
    return dtype in fp8s


def _random_tensor(shape, *, dtype, device) -> torch.Tensor:
    """Like ``torch.randn(shape, dtype=dtype)`` but supports fp8 by
    drawing samples in bf16 then casting. ``torch.randn`` raises
    ``NotImplementedError: normal_kernel_cuda for Float8_e4m3fn``
    so we route fp8 through a bf16 buffer.
    """
    if _is_fp8(dtype):
        buf = torch.randn(shape, dtype=torch.bfloat16, device=device)
        return buf.to(dtype)
    return torch.randn(shape, dtype=dtype, device=device)


def _compute_dtype_for(kv_dtype: torch.dtype) -> torch.dtype:
    """Match the production driver: fp8 KV cache reads under bf16
    compute. The FlashInfer fa2/fa3 backends don't support fp8 output,
    and the 70B-fp8 driver pairs fp8 KV with bf16 Q (see driver.py:774).
    Use this to derive the right ``q_data_type`` for the probes from a
    KV-side dtype.
    """
    return torch.bfloat16 if _is_fp8(kv_dtype) else kv_dtype


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
    """Empty-kernel launch overhead — used as an anchor for ``decode_launch_us``
    when the OLS-fitted intercept is implausibly high (see ``calibrate()``)."""
    import triton

    @triton.jit
    def _empty():
        pass

    return _time_event_us(lambda: _empty[(1,)](), n, device=device)


def measure_merge_us(
    device: torch.device,
    *,
    num_heads: int = 32,
    head_dim: int = 128,
    n: int = 50,
    dtype: torch.dtype = torch.float16,
    K_pairs: tuple[int, ...] = (8, 2048),
) -> tuple[float, float]:
    """Linear regression of ``merge_state_in_place`` time vs row count.

    Returns ``(launch_us, us_per_row)``: the intercept and slope of the
    best-fit line ``time_us ≈ launch_us + us_per_row × K``. At small K
    the merge is launch-dominated (tens of bytes moved); at large K the
    bandwidth term dominates. The two-point fit captures both regimes
    so the cost model can scale merge cost with B·K at runtime.

    ``num_heads`` / ``head_dim`` are deployment-specific (post-TP-shard);
    the autotune script passes the per-rank values so the fitted slope
    reflects the actual merge work this deployment does.
    """
    from flashinfer import merge_state_in_place

    xs: list[float] = []
    ys: list[float] = []
    for K in K_pairs:
        v_a = _random_tensor((K, num_heads, head_dim), dtype=dtype, device=device)
        s_a = torch.randn(K, num_heads, dtype=torch.float32, device=device)
        v_b = _random_tensor((K, num_heads, head_dim), dtype=dtype, device=device)
        s_b = torch.randn(K, num_heads, dtype=torch.float32, device=device)
        t = _time_event_us(
            lambda: merge_state_in_place(v_a, s_a, v_b, s_b),
            n,
            device=device,
        )
        xs.append(float(K))
        ys.append(t)
    # Two-point line fit (or n-point if more pairs are given). Use the
    # closed-form OLS slope/intercept so calibrate.py stays dependency-free.
    n_pts = len(xs)
    mx = sum(xs) / n_pts
    my = sum(ys) / n_pts
    var = sum((x - mx) ** 2 for x in xs)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = cov / var if var > 0 else 0.0
    intercept = my - slope * mx
    return max(0.0, intercept), max(0.0, slope)


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
        # Eight samples spanning two regimes:
        #   - L2-resident / launch-dominated: (32, 128), (32, 256),
        #     (64, 128), (64, 256) — the working set fits in the H100's
        #     ~50 MB L2, so the kernel's effective BW exceeds the
        #     all-HBM rate and the intercept reflects warm-pipeline
        #     launch cost (not the empty-kernel measurement). This is
        #     the regime real beam-search decode cells (K=32-64 with
        #     short tails) actually live in.
        #   - HBM-bound: (256, 256), (256, 1024), (2048, 64), (2048, 256)
        #     — large BS×L_kv pushes past L2 and the slope tracks HBM
        #     bandwidth.
        # OLS over both regimes gives a slope and intercept biased
        # toward the small-BS / short-tail behavior that matters for
        # the picker on multi_doc_qa-style cells.
        (32, 128), (32, 256), (64, 128), (64, 256),
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

    # Probe in production pairing: fp8 KV cache reads under bf16 Q.
    q_dtype = _compute_dtype_for(dtype)
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
        kv = _random_tensor(
            (total_pages, 2, page_size, num_kv_heads, head_dim),
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
        # Pass q/kv dtype so the wrapper's plan path dispatches the
        # matching kernel; otherwise it defaults to fp16 and rejects the
        # bf16/fp8 inputs we hand it on the run() call.
        wrapper.plan(
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            q_data_type=q_dtype,
            kv_data_type=dtype,
        )
        q = _random_tensor(
            (BS, num_qo_heads, head_dim), dtype=q_dtype, device=device,
        )
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
    """Wall time of a single prefill call with T queries × L_kv KV tokens.

    NOTE: this measurement is overhead-dominated at typical probe shapes
    (the wrapper.run Python path + CUDA dispatch is ~95 µs on H100,
    swamping the ~1 µs of actual per-tile compute we'd like to extract).
    ``autotune.autotune()`` therefore resets ``per_tile_us`` to the
    hardcoded defaults in ``Coefficients`` before fitting, so the
    production picker uses {16: 0.6, 64: 1.4, 128: 2.4} regardless of
    what this probe returns. Kept here so the cache key shape is stable
    across calibrate runs.
    """
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    # Probe in the production pairing: fp8 KV is read under bf16 Q (the
    # fa2/fa3 backends don't support fp8 output, so fp8/fp8 would crash).
    # See _compute_dtype_for.
    q_dtype = _compute_dtype_for(dtype)
    num_pages = L_kv // page_size
    kv = _random_tensor(
        (num_pages, 2, page_size, num_kv_heads, head_dim),
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
        q_data_type=q_dtype,
        kv_data_type=dtype,
    )
    q = _random_tensor((T, num_qo_heads, head_dim), dtype=q_dtype, device=device)
    return _time_event_us(lambda: wrapper.run(q, kv), n, device=device)


def measure_per_tile_per_kv_us(
    device: torch.device,
    T: int,
    *,
    num_kv_heads: int = 8,
    num_qo_heads: int = 32,
    head_dim: int = 128,
    page_size: int = 16,
    n: int = 50,
    dtype: torch.dtype = torch.float16,
) -> float:
    """Two-point fit of the per-(kv-token) slope at fixed q_tile (T queries).

    Times prefill at L_kv ∈ {PER_TILE_L_KV_REF, PER_TILE_L_KV_LONG}, then
    fits slope = (t_long − t_ref) / (L_kv_long − L_kv_ref). The intercept
    (launch + per-call overhead) cancels. The CTA-granularity wave model
    consumes this slope as the per-(T_Q,T_KV) MMA cost (via
    τ_elem = slope · cta_tile_kv).
    """
    t_ref = measure_per_tile_us(
        device, T,
        num_kv_heads=num_kv_heads, num_qo_heads=num_qo_heads,
        head_dim=head_dim, L_kv=PER_TILE_L_KV_REF, page_size=page_size,
        n=n, dtype=dtype,
    )
    t_long = measure_per_tile_us(
        device, T,
        num_kv_heads=num_kv_heads, num_qo_heads=num_qo_heads,
        head_dim=head_dim, L_kv=PER_TILE_L_KV_LONG, page_size=page_size,
        n=n, dtype=dtype,
    )
    dt = max(0.0, t_long - t_ref)
    return dt / float(PER_TILE_L_KV_LONG - PER_TILE_L_KV_REF)


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
    merge_launch, merge_bw_per_row = measure_merge_us(device)
    launch = measure_launch_us(device)

    # Per-dtype calibration sweep. Each dtype runs its own per_tile and
    # decode-per-kv probes; rates differ across dtypes because the
    # cascade kernel's elemental MMA throughput and the decode kernel's
    # bw/dequant balance depend on the KV format. fp8 is included only
    # if the FlashInfer probe path accepts it on this build — otherwise
    # the dtype is skipped and the cost model falls back to fp16 rates
    # for that workload (still works, just less accurate).
    per_tile_us_by_dtype: dict[str, dict[int, float]] = {}
    per_tile_per_kv_us_by_dtype: dict[str, dict[int, float]] = {}
    decode_per_token_by_dtype: dict[str, float] = {}
    decode_launch_anchor: float | None = None
    for dtype_label, dtype in _default_probe_dtypes():
        if verbose:
            print(f"[calibrate] probing dtype={dtype_label}")
        try:
            per_tile_us_by_dtype[dtype_label] = {
                T: measure_per_tile_us(device, T, dtype=dtype)
                for T in (T_SMALL, *T_LARGE_CHOICES)
            }
            per_tile_per_kv_us_by_dtype[dtype_label] = {
                T: measure_per_tile_per_kv_us(device, T, dtype=dtype)
                for T in (T_SMALL, *T_LARGE_CHOICES)
            }
            raw_decode_launch, decode_per_kv_raw = measure_decode_per_kv_us(
                device, dtype=dtype,
            )
        except Exception as e:
            if verbose:
                print(
                    f"  skipping dtype={dtype_label}: "
                    f"{type(e).__name__}: {e}"
                )
            continue
        # ``decode_launch_us`` represents the EXTRA per-call cost the
        # picker pays when DEC_TAIL adds a decode kernel call vs
        # FUSED's single-fused-prefill. Physically that's one CUDA
        # launch + minimal plan/dispatch work — same magnitude as
        # ``launch_us``. Anchor at the empty-kernel measurement when
        # the OLS intercept is implausibly high. The anchored value is
        # dtype-independent (CUDA launch overhead); take the first
        # dtype's measurement as the reference.
        if raw_decode_launch > launch:
            decode_launch_dt = launch
        else:
            decode_launch_dt = raw_decode_launch
        if decode_launch_anchor is None:
            decode_launch_anchor = decode_launch_dt
            if verbose:
                print(
                    f"  raw decode_launch={raw_decode_launch:.2f} → "
                    f"anchored to launch_us={launch:.2f}"
                )
        decode_per_token_by_dtype[dtype_label] = decode_per_kv_raw

    # Guard against the unlikely case where every dtype probe failed —
    # fall back to defaults so the cache write doesn't produce an empty
    # coefficients object.
    if not decode_per_token_by_dtype:
        defaults = Coefficients.defaults()
        per_tile_us_by_dtype = dict(defaults.per_tile_us)
        per_tile_per_kv_us_by_dtype = dict(defaults.per_tile_per_kv_us)
        decode_per_token_by_dtype = dict(defaults.decode_us_per_beam_kv_token)
        decode_launch_anchor = launch

    coeffs = Coefficients(
        B_hbm=measure_B_hbm(device),
        merge_launch_us=merge_launch,
        merge_bw_us_per_row=merge_bw_per_row,
        per_tile_us=per_tile_us_by_dtype,
        per_tile_per_kv_us=per_tile_per_kv_us_by_dtype,
        num_sms=num_sms,
        decode_us_per_beam_kv_token=decode_per_token_by_dtype,
        decode_launch_us=decode_launch_anchor or launch,
    )
    _save(coeffs, path)
    if verbose:
        print(f"[calibrate] wrote {path}")
    return coeffs


def _print_coeffs(c: Coefficients) -> None:
    print(f"  B_hbm                = {c.B_hbm:>10.1f} bytes/µs   ({c.B_hbm / 1e6:.2f} TB/s)")
    print(f"  merge_launch_us      = {c.merge_launch_us:>10.4f}")
    print(f"  merge_bw_us_per_row  = {c.merge_bw_us_per_row:>10.6f}")
    print(f"  decode_launch_us     = {c.decode_launch_us:>10.2f}")
    for dtype_label in sorted(c.per_tile_us.keys()):
        inner = c.per_tile_us[dtype_label]
        for T in sorted(inner.keys()):
            print(f"  per_tile[{dtype_label:>9s}][{T:3d}] = {inner[T]:>10.2f} µs")
    for dtype_label in sorted(c.decode_us_per_beam_kv_token.keys()):
        rate = c.decode_us_per_beam_kv_token[dtype_label]
        print(f"  decode_us_per_beam[{dtype_label:>9s}] = {rate:>10.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Calibrate bs_kernel cost-model coefficients.")
    ap.add_argument("--force", action="store_true", help="re-run probes and overwrite cache")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    c = calibrate(args.device, force=args.force, verbose=True)
    _print_coeffs(c)

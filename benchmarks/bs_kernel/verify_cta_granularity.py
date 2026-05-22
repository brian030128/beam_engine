"""Verify the CTA-granularity wave-quant fix on cost_shared_batch.

Compares the OLD elemental-sub-tile compute formula and the NEW
CTA-granularity formula against the observed L1-marginal Δ from
``probe_cascade_thin_q.py``.

For each (K, L_p, L_tail) probe shape (T derived from L0's packed-Q):
  * compute_us_OLD(L0)     vs compute_us_OLD(L0+L1)  → predicted_Δ_old
  * compute_us_NEW(L0)     vs compute_us_NEW(L0+L1)  → predicted_Δ_new
  * observed Δ from the probe run

The wave-quant knee should appear in NEW but not in OLD.

Usage:
    uv run python benchmarks/bs_kernel/verify_cta_granularity.py
"""

from __future__ import annotations
import math

from beam_engine.methods.bs_kernel.cost_model import (
    Coefficients, T_SMALL, WorkloadShape, cost_shared_batch,
)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# Levels: list of (g_count, beams_per_g_in_model_units, L_kv, bytes_per_kv).
# Mirrors the (g_count, beams_per_g, kv_tokens, bytes_per_kv) tuples that
# _per_prompt_levels produces.
def _route_T(beams_per_g: int, *, pool_count: int, t_large: int) -> int:
    """Mirror _tile_breakdown's pool routing."""
    if pool_count == 1:
        return t_large
    return T_SMALL if beams_per_g <= T_SMALL else t_large


# FlashInfer split-K heuristic constants (cascade.py).
MIN_KV_LEN_FOR_SPLIT = 1024
MIN_KV_CHUNK_TOKENS = 128
PAGE_SIZE = 16


def _split_k_chunks_for_level(
    n_q_tiles_level: int,
    L_kv: int,
    *,
    num_kv_heads: int,
    num_sms: int,
) -> int:
    """Mirror FlashInfer's per-level split-K decision (cascade.py:1080-1173).

    target_tuples = num_sms / num_kv_heads; if level's q_tiles already
    saturates, no split-K. Else split L_kv so each (q_tile, kv_chunk)
    becomes one CTA and CTA count approaches target_tuples per kv_head.
    """
    if L_kv < MIN_KV_LEN_FOR_SPLIT:
        return 1
    target_tuples = max(num_sms // max(1, num_kv_heads), 1)
    if n_q_tiles_level >= target_tuples:
        return 1
    target_chunks = max(target_tuples // n_q_tiles_level, 1)
    chunk_size = max(_ceil_div(L_kv, target_chunks), MIN_KV_CHUNK_TOKENS)
    chunk_size = ((chunk_size + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    return max(1, _ceil_div(L_kv, chunk_size))


def compute_us_old(
    levels, *, t_large: int, num_kv_heads: int, pool_count: int, c: Coefficients,
) -> float:
    """Elemental-sub-tile wave model (pre-fix). M_s = Σ n_q × kv_chunks;
    waves = ⌈M_s/N_SM⌉; compute = waves × τ_elem.

    Note: the OLD code computes one global ceil over M_s regardless of
    pool — that's also a bug, but we keep it here for fidelity to the
    pre-fix model. (See cost_shared_batch on the main branch.)"""
    slope = c.per_tile_per_kv_us.get(t_large, 0.0)
    tau_elem = slope * c.cta_tile_kv if slope > 0 else 0.0
    M_s = 0
    for g_count, beams_per_g, L_kv, _bpkv in levels:
        if L_kv == 0:
            continue
        T = _route_T(beams_per_g, pool_count=pool_count, t_large=t_large)
        n_q_tiles = g_count * max(1, _ceil_div(beams_per_g, T))
        kv_chunks = max(1, _ceil_div(L_kv, c.cta_tile_kv))
        M_s += n_q_tiles * kv_chunks
    waves = max(1, _ceil_div(M_s, c.num_sms))
    return waves * tau_elem


def compute_us_new(
    levels, *, t_large: int, num_kv_heads: int, pool_count: int, c: Coefficients,
) -> float:
    """CTA-granularity wave model (post-fix), with FlashInfer split-K
    modeled per level.

    Per pool: total CTAs = Σ_level n_q_tiles × num_chunks_l × num_kv_heads.
    Per-CTA τ in level l: ⌈(L_kv/num_chunks_l)/cta_tile_kv⌉ × τ_elem.
    Wave = ⌈CTAs/N_SM⌉ once per pool; compute = waves × max_l(τ_CTA_l).
    """
    slope = c.per_tile_per_kv_us.get(t_large, 0.0)
    tau_elem = slope * c.cta_tile_kv if slope > 0 else 0.0
    pool_ctas: dict[int, int] = {}
    pool_max_tau: dict[int, float] = {}
    for g_count, beams_per_g, L_kv, _bpkv in levels:
        if L_kv == 0:
            continue
        T = _route_T(beams_per_g, pool_count=pool_count, t_large=t_large)
        n_q_tiles = g_count * max(1, _ceil_div(beams_per_g, T))
        # Split-K: each q_tile becomes num_chunks CTAs scanning L_kv/num_chunks each.
        num_chunks = _split_k_chunks_for_level(
            n_q_tiles, L_kv,
            num_kv_heads=num_kv_heads, num_sms=c.num_sms,
        )
        chunk_L_kv = _ceil_div(L_kv, num_chunks)
        n_ctas = n_q_tiles * num_chunks * num_kv_heads
        kv_iters_per_cta = max(1, _ceil_div(chunk_L_kv, c.cta_tile_kv))
        tau_cta = kv_iters_per_cta * tau_elem
        pool_ctas[T] = pool_ctas.get(T, 0) + n_ctas
        if tau_cta > pool_max_tau.get(T, 0.0):
            pool_max_tau[T] = tau_cta
    total = 0.0
    for T, n_ctas in pool_ctas.items():
        waves = max(1, _ceil_div(n_ctas, c.num_sms))
        total += waves * pool_max_tau.get(T, 0.0)
    return total


def derive_T(K: int, gqa_group_size: int) -> int:
    """Match probe_cascade_thin_q.py's L0-based T choice."""
    packed = K * gqa_group_size
    if packed <= T_SMALL:
        return T_SMALL
    elif packed <= 64:
        return 64
    else:
        return 128


def main():
    # Use the H100 calibrated coefficients written by the recalibration
    # job. Read the JSON cache directly so this runs without a GPU.
    import json
    from pathlib import Path
    from beam_engine.methods.bs_kernel.calibrate import _from_payload
    cache = Path.home() / ".cache/beam_engine/coeffs-NVIDIA_H100_80GB_HBM3.json"
    if cache.exists():
        c = _from_payload(json.loads(cache.read_text()))
        print(f"[verify] loaded H100 calibrated coefficients from {cache}")
    else:
        c = Coefficients(num_sms=132, per_tile_per_kv_us={
            16: 0.0141, 64: 0.0144, 128: 0.0144,
        })
        print("[verify] H100 cache not found, using fallback defaults")

    # (K, L_p, L_tail, T_num1_observed_us, T_num2_observed_us, Δ_observed_us)
    # — from probe_cascade_thin_q.py run (T=128 forced for all rows).
    shapes = [
        (16,  8192,   128,  46.34,  56.87,  10.53),
        (32,  8192,   128,  59.00,  79.60,  20.61),
        (32,  8192,   256,  59.29,  80.28,  21.00),
        (32,  32768,  128, 151.08, 223.10,  72.02),
        (32,  32768,  256, 151.16, 223.65,  72.48),
        (64,  8192,   128,  89.17, 134.67,  45.50),
        (64,  32768,  128, 269.52, 423.26, 153.74),
    ]

    num_qo_heads, num_kv_heads = 64, 4
    gqa = num_qo_heads // num_kv_heads
    bytes_per_kv = 2 * num_kv_heads * 128 * 2  # 2(K+V) × kv_heads × head_dim × bf16

    print(f"GQA group size = {gqa};  num_sms = {c.num_sms};  τ_elem = "
          f"{c.per_tile_per_kv_us[128] * c.cta_tile_kv:.3f} µs")
    print()
    print("=== Standalone formulas (compute_us only): OLD vs NEW Δ ===")
    header = (f"{'K':>3} {'L_p':>6} {'L_tail':>6} {'T':>4} | "
              f"{'obs Δ':>7} | "
              f"{'OLD Δ':>7} {'OLD err':>7} | "
              f"{'NEW Δ':>7} {'NEW err':>7}")
    print(header)
    print("-" * len(header))

    err_sum_old = 0.0
    err_sum_new = 0.0
    for (K, L_p, L_tail, _, _, obs_dt) in shapes:
        T = derive_T(K, gqa)
        # PHYSICAL convention: packed-Q = K*gqa.
        levels_A = [(1, K * gqa, L_p, bytes_per_kv)]
        levels_B = [(1, K * gqa, L_p, bytes_per_kv),
                    (K, 1 * gqa, L_tail, bytes_per_kv)]

        old_A = compute_us_old(levels_A, t_large=T, num_kv_heads=num_kv_heads, pool_count=1, c=c)
        old_B = compute_us_old(levels_B, t_large=T, num_kv_heads=num_kv_heads, pool_count=1, c=c)
        new_A = compute_us_new(levels_A, t_large=T, num_kv_heads=num_kv_heads, pool_count=1, c=c)
        new_B = compute_us_new(levels_B, t_large=T, num_kv_heads=num_kv_heads, pool_count=1, c=c)

        d_old = old_B - old_A
        d_new = new_B - new_A
        e_old = d_old - obs_dt
        e_new = d_new - obs_dt
        err_sum_old += abs(e_old)
        err_sum_new += abs(e_new)

        print(f"{K:>3} {L_p:>6} {L_tail:>6} {T:>4} | "
              f"{obs_dt:>7.2f} | "
              f"{d_old:>7.2f} {e_old:>+7.2f} | "
              f"{d_new:>7.2f} {e_new:>+7.2f}")
    print(f"\n MAE  OLD: {err_sum_old/len(shapes):>7.2f}  "
          f"NEW: {err_sum_new/len(shapes):>7.2f}  µs")

    # End-to-end check: actual refactored cost_shared_batch (depth=2).
    # We have no observed T for Config A through this API (it asserts
    # depth>=2), so we only compare Config B predicted vs observed T_num2.
    print("\n=== Refactored cost_shared_batch (depth=2): total predicted vs observed T_num2 ===")
    h2 = f"{'K':>3} {'L_p':>6} {'L_tail':>6} | {'obs T_num2':>10} | {'pred':>8} | {'err':>8}"
    print(h2)
    print("-" * len(h2))
    for (K, L_p, L_tail, _, obs_t2, _) in shapes:
        w = WorkloadShape(
            K=K, L_p=L_p, suffix_lens=[L_tail] * K,
            num_kv_heads=num_kv_heads, head_dim=128,
            bytes_per_kv=bytes_per_kv,
            num_qo_heads=num_qo_heads,
        )
        pred = cost_shared_batch([w], c, depth=2, pool_count=1, t_large=derive_T(K, gqa))
        err = pred - obs_t2
        print(f"{K:>3} {L_p:>6} {L_tail:>6} | {obs_t2:>10.2f} | {pred:>8.2f} | {err:>+8.2f}")


if __name__ == "__main__":
    main()

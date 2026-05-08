"""Unified cost model for the beam-search-specialized attention kernel.

Picks one of several attention strategies per decode step. The choice
space and the per-strategy cost expressions correspond to the four
decisions justified in the design plan:

  D1: share-prefix-or-not    Strategy ∈ {SHARED, PER_BEAM}
  D2: cascade depth (1-3)    when SHARED
  D3: pool count (1 or 2)    when SHARED
  D4: T_large per pool       when SHARED

All cost expressions yield estimated per-layer wall time in microseconds.
The driver picks the minimum-cost strategy each step and dispatches the
matching wrapper call.

Cost terms in the current model:
  * HBM bandwidth   (bytes_loaded / B_hbm)
  * SM-occupancy compute  (waves × per_tile_us[T])
  * Autotuned slack       (share_extra_us, dual_pool_extra_us)

Launch / sync / merge overheads are intentionally *not* modeled: for
beam-search workloads the per-step kernel work dwarfs those constant
terms, so the picker is dominated by SM utilization. The fields
``launch_us``, ``sync_us``, ``merge_us`` remain on ``Coefficients`` for
calibration cache compatibility but no longer enter the cost.

Coefficients (`Coefficients`) come from per-device calibration
(see `calibrate.py`); on first use, fall back to plausible RTX-class
defaults so the model is always callable for unit tests / dev work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Strategy(Enum):
    PER_BEAM = "per_beam"           # plain paged decode, no sharing
    SHARED_2L_1POOL = "shared_2l_1pool"
    SHARED_2L_2POOL = "shared_2l_2pool"
    SHARED_3L_1POOL = "shared_3l_1pool"
    SHARED_3L_2POOL = "shared_3l_2pool"
    # Hybrid: prefill kernel for prefix (and intermediate, at 3L) +
    # paged-decode kernel for the per-beam tail + merge_state_in_place.
    # Decode kernel is purpose-built for CTA_Q=1 so the per-beam level
    # has 0% padding (vs ~94-98% under prefill at T∈{16,64,128}).
    SHARED_2L_DEC_TAIL = "shared_2l_dec_tail"
    SHARED_3L_DEC_TAIL = "shared_3l_dec_tail"


# T values the kernel JIT-compiles support for. Cost model picks among
# these; calibration measures `per_tile_us` for each.
T_LARGE_CHOICES = (64, 128)
T_SMALL = 16


@dataclass
class Coefficients:
    """Device-calibrated cost-model coefficients (see calibrate.py).

    All times in microseconds. ``B_hbm`` in bytes/microsecond.
    Defaults are conservative RTX-PRO-6000-class numbers so the model
    is callable for tests when no calibration has been done.
    """
    B_hbm: float = 1_000_000.0          # ~1 TB/s (1e6 bytes/µs = 1e12 B/s)
    launch_us: float = 6.0              # empty-kernel launch overhead
    sync_us: float = 4.0                # cudaStreamSynchronize cost
    # merge_state_in_place is a small in-place kernel — H100 measurement
    # at B*K=2048 rows × num_qo_heads × head_dim is ~2 µs. The previous
    # default (8) was an over-estimate that biased the picker against
    # DEC_TAIL strategies which rely on this kernel.
    merge_us: float = 2.0

    # Per-tile time table (µs per CTA-tile).
    # Indexed by T_large ∈ T_LARGE_CHOICES + the small tile T_SMALL.
    per_tile_us: dict[int, float] = field(default_factory=lambda: {
        16: 0.6,
        64: 1.4,
        128: 2.4,
    })

    # Compute-time per row in a tile (rough; refined by per_tile_us
    # which already absorbs Q-side cost). Used only by the per-beam
    # path which doesn't tile.
    per_beam_us_per_kv_token: float = 0.0008

    # ------------------------------------------------------------------
    # SM count for wave-occupancy modeling. Tiles within a pool run in
    # parallel across SMs; the per-tile compute scales with
    # ceil(tile_count / num_sms), not tile_count. Calibrated from
    # ``torch.cuda.get_device_properties().multi_processor_count``.
    # A6000 = 84, RTX 6000 Ada = 142, H100 = 132.
    num_sms: int = 84

    # Auto-tuned (filled by autotune.py). These two parameters absorb
    # device-specific effects the closed-form expressions above don't
    # capture:
    #   * `share_extra_us`     - flat per-layer overhead added to
    #     T_shared. Captures unmodeled merge-launch cost, plus the
    #     L2-cache reuse advantage PER_BEAM gets at small K (the
    #     bandwidth model treats reads as HBM-bound; reality has
    #     cross-beam KV reuse hot in L2). Tuned upward → favors
    #     PER_BEAM more.
    #   * `dual_pool_extra_us` - flat extra overhead added to T_shared
    #     when pool_count=2. Captures SM-occupancy effects beyond the
    #     wave-count term — even when both pools fit in 1 wave each,
    #     the inter-pool sync + extra launch is real overhead.
    #     Tuned upward → favors single-pool dispatch.
    # Both default to 0; pass `autotune=True` to calibrate to fit them.
    share_extra_us: float = 0.0
    dual_pool_extra_us: float = 0.0

    # Decode-kernel per-(beam, kv-token) cost for the DEC_TAIL strategies.
    # Decode kernel is purpose-built for CTA_Q=1 and is bandwidth-bound
    # at our shapes — compute and bandwidth overlap rather than add, so
    # the cost expression uses max(bw_us, work_us) not bw_us + work_us.
    # Calibrated by `calibrate.measure_decode_per_kv_us`.
    decode_us_per_beam_kv_token: float = 0.0008
    # Per-step constant: post-launch / kernel-init cost for the decode
    # kernel. With the bs_kernel driver's plan_info cache the call is
    # essentially a kernel launch (~2 µs) per layer, no scheduling work.
    decode_launch_us: float = 2.0

    @classmethod
    def defaults(cls) -> "Coefficients":
        return cls()


@dataclass
class WorkloadShape:
    """Per-step workload signature the cost model needs.

    ``L_p`` is the LCA prefix length in *tokens* (not pages). ``suffix_lens``
    is per-beam unique-tail length in tokens. ``intermediate`` describes
    a 3-level fork structure when one exists, else None.
    """
    K: int
    L_p: int
    suffix_lens: list[int]
    num_kv_heads: int
    head_dim: int
    bytes_per_kv: int          # 2 (k+v) × num_kv_heads × head_dim × dtype_bytes
    intermediate: Optional["IntermediateShape"] = None


@dataclass
class IntermediateShape:
    """When the cascade can also do a 3-level decomposition.

    ``groups`` is a list of ``(beams_in_group, inter_len_tokens)`` per
    intermediate group — non-uniform sizes and run lengths are allowed.
    Singleton groups (1 beam) contribute no bandwidth savings but are
    admitted so every query stays in a group at every level.
    """
    groups: list[tuple[int, int]]


# ---------------------------------------------------------------------------
# Cost expressions
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _bytes_to_us(n_bytes: int, c: Coefficients) -> float:
    return n_bytes / c.B_hbm


def _level_tiles_and_bytes(
    levels: list[tuple[int, int, int, int]],
    *,
    pool_count: int,
    t_large: int,
) -> tuple[int, int, int]:
    """Sum (large_tiles, small_tiles, bytes_loaded) over a list of
    cascade levels. Each level is (group_count, beams_per_group,
    kv_tokens, bytes_per_kv).
    """
    total_t_large_tiles = 0
    total_t_small_tiles = 0
    bytes_loaded = 0
    for g_count, beams_per_g, kv_tokens, bytes_per_kv in levels:
        if kv_tokens == 0:
            continue
        packed = beams_per_g
        if pool_count == 1:
            tiles_per_group = max(1, _ceil_div(packed, t_large))
            total_t_large_tiles += g_count * tiles_per_group
        else:
            if packed <= T_SMALL:
                tiles_per_group = max(1, _ceil_div(packed, T_SMALL))
                total_t_small_tiles += g_count * tiles_per_group
            else:
                tiles_per_group = max(1, _ceil_div(packed, t_large))
                total_t_large_tiles += g_count * tiles_per_group
        bytes_loaded += g_count * kv_tokens * bytes_per_kv
    return total_t_large_tiles, total_t_small_tiles, bytes_loaded


def _per_prompt_levels(w: WorkloadShape, depth: int) -> list[tuple[int, int, int, int]]:
    """Build the (g_count, beams_per_group, kv_tokens, bytes_per_kv) level
    tuples for one prompt's contribution to a cascade plan.

    For depth=3 we emit one tuple *per intermediate group* (non-uniform
    sizes and run lengths supported). The aggregator in
    ``_level_tiles_and_bytes`` sums tile counts and bytes across all the
    tuples regardless of how many groups the level contains.

    ``suffix_lens`` in the workload reflects the layout produced with
    ``max_levels=3`` — i.e., per-beam tails *after* the intermediate
    run. When pricing depth=2 against the same workload, those
    intermediate pages would otherwise be missing from the bandwidth
    term, so we fold them back into the average suffix length here.
    """
    levels: list[tuple[int, int, int, int]] = []
    # Level 0: 1 group of K beams sharing L_p tokens.
    levels.append((1, w.K, w.L_p, w.bytes_per_kv))
    avg_suffix_post = sum(w.suffix_lens) / max(1, len(w.suffix_lens))
    if depth == 3:
        assert w.intermediate is not None, "depth=3 needs intermediate shape"
        for beams_in_grp, inter_len in w.intermediate.groups:
            levels.append((1, beams_in_grp, inter_len, w.bytes_per_kv))
        avg_suffix = avg_suffix_post
    else:
        # depth=2: intermediate pages collapse into per-beam tails. Add
        # the per-beam-average intermediate length so the bandwidth term
        # accounts for K independent reads of those pages.
        if w.intermediate is not None:
            extra_per_beam = sum(
                g[0] * g[1] for g in w.intermediate.groups
            ) / max(1, w.K)
            avg_suffix = avg_suffix_post + extra_per_beam
        else:
            avg_suffix = avg_suffix_post
    # Last level: K groups of 1 beam each.
    levels.append((w.K, 1, int(round(avg_suffix)), w.bytes_per_kv))
    return levels


def cost_per_beam_batch(workloads: list[WorkloadShape], c: Coefficients) -> float:
    """Plain paged decode for B prompts in one fused kernel launch.

    Each beam loads (L_p_b + suffix_len) of K/V. The single batched
    launch overhead is no longer modeled (negligible vs the kernel's
    HBM read for beam-search shapes).
    """
    total_kv_bytes = 0
    for w in workloads:
        total_kv_bytes += sum(w.L_p + s for s in w.suffix_lens) * w.bytes_per_kv
    return _bytes_to_us(total_kv_bytes, c)


def cost_shared_batch(
    workloads: list[WorkloadShape],
    c: Coefficients,
    *,
    depth: int,
    pool_count: int,
    t_large: int,
    fused_merge: bool = False,
) -> float:
    """Cost of the shared-prefix path running B prompts in one cascade
    launch.

    All prompts use the same ``(depth, pool_count, t_large)`` — that's
    the choice the picker has to make for the fused kernel. Within
    each prompt, the per-level tile counts are the same as the B=1
    case. Across prompts, tile counts SUM, then wave count is computed
    against ``num_sms`` ONCE across the batch — this is the cross-prompt
    wave-occupancy effect that B=1 reasoning misses.

    depth ∈ {2, 3}. depth=3 requires every prompt's workload have an
    intermediate (else this candidate is invalid for the batch and the
    caller should skip it).

    ``fused_merge`` is retained as an API-stable no-op kwarg; merge cost
    is no longer modeled (see module docstring).
    """
    del fused_merge  # no longer modeled
    assert depth in (2, 3), depth
    assert pool_count in (1, 2), pool_count
    assert t_large in T_LARGE_CHOICES, t_large
    if depth == 3:
        for w in workloads:
            assert w.intermediate is not None, (
                "depth=3 batch needs intermediate on every prompt"
            )

    # Aggregate tile counts and bytes across all prompts.
    total_large = 0
    total_small = 0
    total_bytes = 0
    for w in workloads:
        levels = _per_prompt_levels(w, depth)
        large, small, b = _level_tiles_and_bytes(
            levels, pool_count=pool_count, t_large=t_large,
        )
        total_large += large
        total_small += small
        total_bytes += b

    bw_us = _bytes_to_us(total_bytes, c)
    # Cross-batch wave count: critical for batched decoding — at B=1 the
    # per-beam tiles fit in 1 wave on H100; at B=8 they may need 4 waves
    # at T=64 (and fewer at T=16, which is what tips 2-pool to win).
    waves_large = max(1, _ceil_div(total_large, c.num_sms)) if total_large > 0 else 0
    waves_small = max(1, _ceil_div(total_small, c.num_sms)) if total_small > 0 else 0
    compute_us = (
        waves_large * c.per_tile_us[t_large]
        + waves_small * c.per_tile_us[T_SMALL]
    )

    # Launch / sync / merge terms have been removed from the model — for
    # beam-search shapes the kernel work dwarfs them. SM utilization
    # (the wave-count term above) is what drives the picker. The
    # autotuned slack still distinguishes 1-pool vs 2-pool because pool
    # routing has second-order effects (occupancy, sync stalls) the
    # closed-form wave count doesn't capture.
    extra_us = c.share_extra_us
    if pool_count == 2:
        extra_us += c.dual_pool_extra_us

    return bw_us + compute_us + extra_us


def _per_prompt_levels_no_tail(
    w: WorkloadShape, depth: int,
) -> list[tuple[int, int, int, int]]:
    """Like ``_per_prompt_levels`` but omits the per-beam tail level.

    Used by the DEC_TAIL strategies: only the prefix (and intermediate at
    depth=3) levels are priced as prefill tiles; the tail goes through
    the decode kernel and is priced separately by
    ``cost_decode_tail_batch``.
    """
    levels: list[tuple[int, int, int, int]] = []
    levels.append((1, w.K, w.L_p, w.bytes_per_kv))
    if depth == 3:
        assert w.intermediate is not None, "depth=3 needs intermediate shape"
        for beams_in_grp, inter_len in w.intermediate.groups:
            levels.append((1, beams_in_grp, inter_len, w.bytes_per_kv))
    return levels


def cost_decode_tail_batch(
    workloads: list[WorkloadShape], c: Coefficients,
) -> float:
    """Cost of the per-beam-tail decode kernel for B prompts × K beams.

    All B*K beams contribute 1 query each against their per-beam tail
    KV. Decode kernel has CTA_Q=1 so there's no tile padding. The
    kernel is memory-bound at our shapes (small per-beam KV, single
    query): MMA work overlaps with HBM reads, so cost is
    ``max(bw_us, work_us) + decode_launch_us``, not the sum.
    """
    total_kv_tokens = 0
    n_beams = 0
    for w in workloads:
        total_kv_tokens += sum(w.suffix_lens)
        n_beams += w.K
    if n_beams == 0:
        return 0.0
    bw_us = total_kv_tokens * w.bytes_per_kv / c.B_hbm
    work_us = total_kv_tokens * c.decode_us_per_beam_kv_token
    return c.decode_launch_us + max(bw_us, work_us)


def cost_dec_tail_batch(
    workloads: list[WorkloadShape],
    c: Coefficients,
    *,
    depth: int,
) -> float:
    """Hybrid: prefix (and optional intermediate) via prefill cascade +
    per-beam tail via paged decode + 1 (or 2) merges.

    All prompts share one ``depth``. depth=3 requires every prompt's
    workload to have an intermediate shape; the picker filters that.
    The prefix-side cost is priced like ``cost_shared_batch`` with
    pool_count=1 and t_large=64 (the common case at K∈{16,64} for
    Llama-3.2-1B: packed_qo at the prefix level is K, not 1, so it's
    typically wide enough to use T=64). The tail-side cost is priced
    by ``cost_decode_tail_batch``.
    """
    assert depth in (2, 3), depth
    if depth == 3:
        for w in workloads:
            assert w.intermediate is not None, (
                "depth=3 batch needs intermediate on every prompt"
            )

    # Prefix (+ intermediate) under prefill kernel, single pool, T=64.
    total_large = 0
    total_bytes = 0
    for w in workloads:
        levels = _per_prompt_levels_no_tail(w, depth)
        large, _small, b = _level_tiles_and_bytes(
            levels, pool_count=1, t_large=64,
        )
        total_large += large
        total_bytes += b
    bw_us = _bytes_to_us(total_bytes, c)
    waves_large = max(1, _ceil_div(total_large, c.num_sms)) if total_large > 0 else 0
    prefix_us = bw_us + waves_large * c.per_tile_us[64] + c.share_extra_us

    # Tail under decode kernel.
    tail_us = cost_decode_tail_batch(workloads, c)

    # Merges: 1 at depth=2 (prefix ⊕ tail), 2 at depth=3 (cascade
    # already merges prefix+inter internally; we add tail merge).
    n_merges = 1 if depth == 2 else 2
    return prefix_us + tail_us + n_merges * c.merge_us


def cost_per_beam(w: WorkloadShape, c: Coefficients) -> float:
    """Plain paged decode for a single prompt — wraps cost_per_beam_batch."""
    return cost_per_beam_batch([w], c)


def cost_shared(
    w: WorkloadShape,
    c: Coefficients,
    *,
    depth: int,
    pool_count: int,
    t_large: int,
    fused_merge: bool = False,
) -> float:
    """Cost of the shared-prefix path for a single prompt — wraps
    ``cost_shared_batch`` with a 1-element list."""
    return cost_shared_batch(
        [w], c,
        depth=depth, pool_count=pool_count, t_large=t_large,
        fused_merge=fused_merge,
    )


# ---------------------------------------------------------------------------
# Picker
# ---------------------------------------------------------------------------


@dataclass
class Pick:
    strategy: Strategy
    estimated_us: float
    t_large: int          # ignored for PER_BEAM
    depth: int            # ignored for PER_BEAM (or =1)
    pool_count: int       # ignored for PER_BEAM (or =1)
    debug: dict           # all candidates' costs, for ablation logging

    @property
    def share(self) -> bool:
        return self.strategy != Strategy.PER_BEAM


def pick_strategy_batch(
    workloads: list[WorkloadShape],
    c: Coefficients,
    *,
    fused_merge: bool = False,
    available_strategies: Optional[set[Strategy]] = None,
) -> Pick:
    """Pick the best strategy for a *batch* of prompts decoded together.

    All prompts share one ``(depth, pool_count, t_large)`` — that's the
    choice forced by a single fused kernel launch. Tile counts sum
    across prompts, so cross-prompt wave overflow tilts the picker
    toward the 2-pool path at large B (where the per-beam tile count
    crosses ``num_sms`` and a separate T_SMALL pool reduces wave count).

    ``depth=3`` is only considered if every prompt has an intermediate
    structure (otherwise the cascade plan can't use a 3-level layout
    consistently — flashinfer wraps a single ``num_levels``).
    """
    debug: dict[str, float] = {}

    # Per-beam baseline.
    pb_cost = cost_per_beam_batch(workloads, c)
    debug["per_beam"] = pb_cost
    candidates: list[tuple[Strategy, float, int, int, int]] = [
        (Strategy.PER_BEAM, pb_cost, 0, 1, 1),
    ]

    all_have_intermediate = all(w.intermediate is not None for w in workloads)

    for depth in (2, 3):
        if depth == 3 and not all_have_intermediate:
            continue
        for pool_count in (1, 2):
            for t_large in T_LARGE_CHOICES:
                cs = cost_shared_batch(
                    workloads, c,
                    depth=depth, pool_count=pool_count,
                    t_large=t_large, fused_merge=fused_merge,
                )
                tag = f"shared_d{depth}_p{pool_count}_t{t_large}"
                debug[tag] = cs
                if depth == 2 and pool_count == 1:
                    s = Strategy.SHARED_2L_1POOL
                elif depth == 2 and pool_count == 2:
                    s = Strategy.SHARED_2L_2POOL
                elif depth == 3 and pool_count == 1:
                    s = Strategy.SHARED_3L_1POOL
                else:
                    s = Strategy.SHARED_3L_2POOL
                candidates.append((s, cs, t_large, depth, pool_count))

    # DEC_TAIL candidates: prefix(/intermediate) prefill + decode tail +
    # merge. T_large doesn't apply to the tail (decode kernel is CTA_Q=1);
    # we record T_large=64 for the prefix-side picker hint.
    for depth in (2, 3):
        if depth == 3 and not all_have_intermediate:
            continue
        cs = cost_dec_tail_batch(workloads, c, depth=depth)
        tag = f"shared_d{depth}_dec_tail"
        debug[tag] = cs
        s = (
            Strategy.SHARED_2L_DEC_TAIL if depth == 2
            else Strategy.SHARED_3L_DEC_TAIL
        )
        candidates.append((s, cs, 64, depth, 1))

    if available_strategies is not None:
        filtered = [cand for cand in candidates if cand[0] in available_strategies]
        if not filtered:
            # Fallback: SHARED_3L_* forced but no intermediate group
            # structure (typical at the first ~16 decode steps before
            # forks settle) collapses to the depth=2 equivalent.
            _fallback = {
                Strategy.SHARED_3L_1POOL: Strategy.SHARED_2L_1POOL,
                Strategy.SHARED_3L_2POOL: Strategy.SHARED_2L_2POOL,
                Strategy.SHARED_3L_DEC_TAIL: Strategy.SHARED_2L_DEC_TAIL,
            }
            expanded = set(available_strategies)
            for s in available_strategies:
                if s in _fallback:
                    expanded.add(_fallback[s])
            filtered = [cand for cand in candidates if cand[0] in expanded]
            if not filtered:
                raise ValueError(
                    "no strategy available after filtering (incl. depth=3 → "
                    "depth=2 fallbacks)"
                )
        candidates = filtered

    best = min(candidates, key=lambda x: x[1])
    return Pick(
        strategy=best[0],
        estimated_us=best[1],
        t_large=best[2],
        depth=best[3],
        pool_count=best[4],
        debug=debug,
    )


def pick_strategy(
    w: WorkloadShape,
    c: Coefficients,
    *,
    fused_merge: bool = False,
    available_strategies: Optional[set[Strategy]] = None,
) -> Pick:
    """Pick the best strategy for a single prompt — wraps
    ``pick_strategy_batch`` with a 1-element list. Existing single-prompt
    callers (driver, oracle_vs_model, autotune) keep working unchanged.
    """
    return pick_strategy_batch(
        [w], c, fused_merge=fused_merge,
        available_strategies=available_strategies,
    )


# ---------------------------------------------------------------------------
# Closed-form crossover helpers (used by ablation A1 to plot the
# share-vs-per-beam boundary analytically).
# ---------------------------------------------------------------------------


def share_breaks_even_at(K: int, c: Coefficients, bytes_per_kv: int) -> int:
    """Return the smallest L_p (in tokens) at which sharing the prefix
    beats per-beam paged decode on bandwidth alone.

    The previous formulation balanced bandwidth savings against
    ``merge_us + launch_us``; with overheads removed from the model,
    sharing always saves bandwidth for K ≥ 2 (any positive L_p) so the
    breakeven is 0. Kept for API stability of ablation A1.
    """
    if K <= 1:
        return math.inf  # type: ignore[return-value]
    return 0

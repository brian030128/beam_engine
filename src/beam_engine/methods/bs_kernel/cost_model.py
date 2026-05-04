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
    merge_us: float = 8.0               # CascadeMerge launch+work, K=8 row scale

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
    """When the cascade can also do a 3-level decomposition."""
    G: int                     # number of fork groups (1 < G < K)
    group_size: int            # K // G
    inter_len_tokens: int      # shared intermediate run, in tokens


# ---------------------------------------------------------------------------
# Cost expressions
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _bytes_to_us(n_bytes: int, c: Coefficients) -> float:
    return n_bytes / c.B_hbm


def cost_per_beam(w: WorkloadShape, c: Coefficients) -> float:
    """Plain paged decode: K independent sequences, each loads (L_p +
    suffix_len_b) tokens of K/V. No merge.
    """
    total_kv = sum(w.L_p + s for s in w.suffix_lens)
    bw_us = _bytes_to_us(total_kv * w.bytes_per_kv, c)
    # Per-beam paged decode launches one kernel.
    return bw_us + c.launch_us


def cost_shared(
    w: WorkloadShape,
    c: Coefficients,
    *,
    depth: int,
    pool_count: int,
    t_large: int,
    fused_merge: bool = False,
) -> float:
    """Cost of the shared-prefix path with given depth/pool/T choices.

    depth ∈ {2, 3}. depth=3 requires w.intermediate is not None.
    pool_count ∈ {1, 2}. pool_count=2 launches both T_large and T_small
    kernels (only relevant when both have non-trivial work).
    t_large ∈ T_LARGE_CHOICES.
    fused_merge=True drops the merge_us term (Phase 2 kernel).
    """
    assert depth in (2, 3), depth
    assert pool_count in (1, 2), pool_count
    assert t_large in T_LARGE_CHOICES, t_large
    if depth == 3:
        assert w.intermediate is not None, "depth=3 needs intermediate shape"

    # --- Per-level shapes ---
    # Level 0: 1 group of K beams sharing L_p tokens.
    # Level mid (when depth=3): G groups of group_size beams sharing
    #   inter_len tokens.
    # Level last: K groups of 1 beam, each with suffix_len_b tokens.
    levels: list[tuple[int, int, int]] = []  # (group_count, beams_per_group, kv_tokens)
    levels.append((1, w.K, w.L_p))
    if depth == 3:
        i = w.intermediate  # type: ignore[assignment]
        levels.append((i.G, i.group_size, i.inter_len_tokens))
    # Last level: each beam has its own suffix_len_b. Approximate as K
    # groups with the average suffix_len for tile-count estimation.
    avg_suffix = sum(w.suffix_lens) / max(1, len(w.suffix_lens))
    levels.append((w.K, 1, int(round(avg_suffix))))

    # --- Tile-count + per-tile cost per level ---
    # A level with G groups of B beams contributes (G × ceil(B*kv_group / T) ×
    # num_kv_heads) work tiles. We use kv_group=1 in the cost model for
    # simplicity (GQA factors in via per_tile_us calibration).
    total_t_large_tiles = 0
    total_t_small_tiles = 0
    bytes_loaded = 0
    for g_count, beams_per_g, kv_tokens in levels:
        if kv_tokens == 0:
            continue
        # Pool routing: per-group "packed Q" = beams_per_g (× kv_group, =1 here).
        packed = beams_per_g
        if pool_count == 1:
            # Everything in T_large; beams_per_g < T_large pads.
            tiles_per_group = max(1, _ceil_div(packed, t_large))
            total_t_large_tiles += g_count * tiles_per_group
        else:
            # Two pools: small-Q groups (packed ≤ T_SMALL) → T_SMALL pool;
            # large-Q groups → T_large pool. In beam search L0 is always
            # large-Q (packed = K), levels with beams_per_g ≤ T_SMALL go
            # small.
            if packed <= T_SMALL:
                tiles_per_group = max(1, _ceil_div(packed, T_SMALL))
                total_t_small_tiles += g_count * tiles_per_group
            else:
                tiles_per_group = max(1, _ceil_div(packed, t_large))
                total_t_large_tiles += g_count * tiles_per_group
        # Bytes loaded for this level: g_count × kv_tokens × bytes_per_kv
        # (K/V loaded once per group's kernel scan).
        bytes_loaded += g_count * kv_tokens * w.bytes_per_kv

    bw_us = _bytes_to_us(bytes_loaded, c)
    compute_us = (
        total_t_large_tiles * c.per_tile_us[t_large]
        + total_t_small_tiles * c.per_tile_us[T_SMALL]
    )

    # Launch overhead: 1 launch for the prefill kernel(s); pool_count=2
    # adds a sync between them.
    launch_us = c.launch_us
    if pool_count == 2:
        launch_us += c.launch_us + c.sync_us

    # Merge: separate CascadeMerge launch unless fused into epilogue.
    merge_us = 0.0 if fused_merge else c.merge_us * (depth - 1)

    return bw_us + compute_us + launch_us + merge_us


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


def pick_strategy(
    w: WorkloadShape,
    c: Coefficients,
    *,
    fused_merge: bool = False,
    available_strategies: Optional[set[Strategy]] = None,
) -> Pick:
    """Enumerate every strategy the cost model considers, return the min.

    ``available_strategies`` lets ablation switches restrict the choice
    space (e.g., force always-shared, force fixed-depth=2).
    """
    debug: dict[str, float] = {}

    # Per-beam baseline.
    pb_cost = cost_per_beam(w, c)
    debug["per_beam"] = pb_cost
    candidates: list[tuple[Strategy, float, int, int, int]] = [
        (Strategy.PER_BEAM, pb_cost, 0, 1, 1),
    ]

    # Shared candidates.
    for depth in (2, 3):
        if depth == 3 and w.intermediate is None:
            continue
        for pool_count in (1, 2):
            for t_large in T_LARGE_CHOICES:
                cs = cost_shared(
                    w, c,
                    depth=depth, pool_count=pool_count,
                    t_large=t_large, fused_merge=fused_merge,
                )
                tag = f"shared_d{depth}_p{pool_count}_t{t_large}"
                debug[tag] = cs
                # Map (depth, pool_count) → Strategy enum.
                if depth == 2 and pool_count == 1:
                    s = Strategy.SHARED_2L_1POOL
                elif depth == 2 and pool_count == 2:
                    s = Strategy.SHARED_2L_2POOL
                elif depth == 3 and pool_count == 1:
                    s = Strategy.SHARED_3L_1POOL
                else:
                    s = Strategy.SHARED_3L_2POOL
                candidates.append((s, cs, t_large, depth, pool_count))

    if available_strategies is not None:
        candidates = [c for c in candidates if c[0] in available_strategies]
        if not candidates:
            raise ValueError("no strategy available after filtering")

    # Pick min.
    best = min(candidates, key=lambda x: x[1])
    return Pick(
        strategy=best[0],
        estimated_us=best[1],
        t_large=best[2],
        depth=best[3],
        pool_count=best[4],
        debug=debug,
    )


# ---------------------------------------------------------------------------
# Closed-form crossover helpers (used by ablation A1 to plot the
# share-vs-per-beam boundary analytically).
# ---------------------------------------------------------------------------


def share_breaks_even_at(K: int, c: Coefficients, bytes_per_kv: int) -> int:
    """Return the smallest L_p (in tokens) at which sharing the prefix
    saves more bandwidth than the merge launch costs. Below this L_p,
    per-beam paged decode wins.

    Derived from cost_share - cost_per_beam ≈ 0 with suffix_len → 0:
      (K-1) × L_p × bytes_per_kv / B_hbm  ≈  merge_us + launch_us
    """
    if K <= 1:
        return math.inf  # type: ignore[return-value]
    rhs_us = c.merge_us + c.launch_us
    bytes_per_token_saved = (K - 1) * bytes_per_kv
    return int(math.ceil(rhs_us * c.B_hbm / bytes_per_token_saved))

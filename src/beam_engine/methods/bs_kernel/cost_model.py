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

Cost terms (paper-spec; see docs/paper_story.md §The cost-model picker):
  * Fused:    C_fused = B_s/β + ⌈M_s/N_SM⌉·τ + ⌈log₂ D_s⌉·µ
  * Dec-tail: C_dt    = C_prefix + ν_decode + B_tail/β
                     + ⌈R·⌈L̄/CTA_K⌉/N_SM⌉·τ_tail_elem
              where R = Σ_b K_b (total decode-kernel CTAs),
              τ_tail_elem = ρ_d · cta_tile_kv_decode, and ν_decode
              is the calibrated per-call decode-kernel launch cost
              (the extra launch DEC_TAIL pays over FUSED). Bandwidth
              and compute are added (no overlap) rather than max'd.

Generic launch / sync overheads, the per-wave overhead floor, and the
prefill per-tile L_kv slope are intentionally *not* modeled in this
simplified form: for beam-search workloads the picker is dominated by
SM-occupancy waves on the prefill side and by bandwidth on the
decode-tail side, so the argmin is stable under those simplifications.

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
    SHARED_4L_1POOL = "shared_4l_1pool"
    SHARED_4L_2POOL = "shared_4l_2pool"
    SHARED_5L_1POOL = "shared_5l_1pool"
    SHARED_5L_2POOL = "shared_5l_2pool"
    SHARED_6L_1POOL = "shared_6l_1pool"
    SHARED_6L_2POOL = "shared_6l_2pool"
    # Hybrid: prefill kernel for prefix (and intermediate levels at 3L+) +
    # paged-decode kernel for the per-beam tail + merge_state_in_place.
    # Decode kernel is purpose-built for CTA_Q=1 so the per-beam level
    # has 0% padding (vs ~94-98% under prefill at T∈{16,64,128}).
    SHARED_2L_DEC_TAIL = "shared_2l_dec_tail"
    SHARED_3L_DEC_TAIL = "shared_3l_dec_tail"
    SHARED_4L_DEC_TAIL = "shared_4l_dec_tail"
    SHARED_5L_DEC_TAIL = "shared_5l_dec_tail"
    SHARED_6L_DEC_TAIL = "shared_6l_dec_tail"
    # Cross-prompt 3-level dec_tail: a SINGLE L0 group of all B*K beams reads
    # the cross-prompt-shared sys ONCE (vs once per per-prompt group in the
    # SHARED_* strategies), L1 = per-prompt bundle groups (fused-cascade
    # prefill), per-beam tail through the decode kernel. Only generated as a
    # candidate when the batch shares a cross-prompt sys prefix
    # (``cross_prompt_lca > 0``); the per-prompt cost functions can't express
    # the cross-prompt group, so it has its own cost (cost_xprompt_dec_tail_batch).
    XPROMPT_DEC_TAIL = "xprompt_dec_tail"


# Default candidate set the production picker considers. Restricted to
# four strategies after the K×B×L_p sweep (results/sweep_paper-*) showed
# every winning pick fell into this set. The 2POOL and {4,5,6}L variants
# are retained in the ``Strategy`` enum so autotune / ablation studies
# can still address them by passing ``available_strategies`` explicitly.
DEFAULT_STRATEGIES: frozenset[Strategy] = frozenset({
    Strategy.PER_BEAM,
    Strategy.SHARED_2L_1POOL,
    Strategy.SHARED_2L_DEC_TAIL,
    Strategy.SHARED_3L_DEC_TAIL,
    # Cross-prompt 3L dec_tail — only contributes a candidate when the batch
    # shares a cross-prompt sys prefix (cross_prompt_lca > 0); a strict no-op
    # otherwise. Replaces SHARED_3L_1POOL (dropped from defaults; per-prompt
    # same-reader 3L never won — see project_bs_kernel_3l_not_faster). The
    # SHARED_3L_1POOL enum member is kept for forced ablations.
    Strategy.XPROMPT_DEC_TAIL,
})


# Helper map: (depth, family, pool_count) → Strategy enum value.
# `family` ∈ {"shared", "dec_tail"}. For "shared", pool_count ∈ {1, 2};
# for "dec_tail", pool_count is ignored (always uses 1-pool prefix).
def _strategy_for(depth: int, family: str, pool_count: int) -> Strategy:
    if family == "shared":
        return Strategy[f"SHARED_{depth}L_{pool_count}POOL"]
    if family == "dec_tail":
        return Strategy[f"SHARED_{depth}L_DEC_TAIL"]
    raise ValueError(f"unknown family: {family!r}")


# T values the kernel JIT-compiles support for. Cost model picks among
# these; calibration measures `per_tile_us` for each.
T_LARGE_CHOICES = (64, 128)
T_SMALL = 16

# Reference L_kv values for the per-tile calibration probes.
# - LONG is used to fit the L_kv slope (per_tile_per_kv_us).
# - REF is the "calibration baseline" embedded in ``per_tile_us``.
# - TINY is used by the multi-tile probe to isolate per-wave scheduling
#   overhead from bandwidth. page_size=16 is the minimum FlashInfer
#   supports for paged KV, so the tile reads almost no HBM.
PER_TILE_L_KV_TINY = 16
PER_TILE_L_KV_REF = 1024
PER_TILE_L_KV_LONG = 8192


# Canonical KV-cache dtype labels used as keys in Coefficients per-dtype
# tables. Calibration probes record rates separately per dtype because
# fp8 KV (compact load, _scaled_mm MMA) has a measurably different
# (compute, HBM) balance than bf16 / fp16; see project memory
# [[project_70b_fp8_picker_recalibrate]].
DTYPE_FP16 = "fp16"
DTYPE_BF16 = "bf16"
DTYPE_FP8_E4M3 = "fp8_e4m3"
DTYPE_FP8_E5M2 = "fp8_e5m2"
DEFAULT_KV_DTYPE = DTYPE_FP16


def canonical_kv_dtype(dtype) -> str:
    """Map a ``torch.dtype`` (or canonical str) to the lookup key used by
    ``Coefficients`` per-dtype tables. Unknown dtypes fall back to fp16
    so legacy callers that pass no dtype keep working.
    """
    if isinstance(dtype, str):
        return dtype
    if dtype is None:
        return DEFAULT_KV_DTYPE
    # Lazy import to keep cost_model importable without torch at module
    # load time (used by test harnesses that stub torch).
    try:
        import torch
    except ImportError:
        return DEFAULT_KV_DTYPE
    mapping = {
        torch.float16:        DTYPE_FP16,
        torch.bfloat16:       DTYPE_BF16,
    }
    if hasattr(torch, "float8_e4m3fn"):
        mapping[torch.float8_e4m3fn] = DTYPE_FP8_E4M3
    if hasattr(torch, "float8_e5m2"):
        mapping[torch.float8_e5m2] = DTYPE_FP8_E5M2
    return mapping.get(dtype, DEFAULT_KV_DTYPE)


@dataclass
class Coefficients:
    """Device-calibrated cost-model coefficients (see calibrate.py).

    All times in microseconds. ``B_hbm`` in bytes/microsecond.
    Defaults are conservative RTX-PRO-6000-class numbers so the model
    is callable for tests when no calibration has been done.
    """
    B_hbm: float = 1_000_000.0          # ~1 TB/s (1e6 bytes/µs = 1e12 B/s)
    # Cascade-merge cost. ``merge_state_in_place`` reads two
    # (B*K, num_qo_heads, head_dim) tensors plus two (B*K, num_qo_heads)
    # softmax-state tensors and writes one of each, so the bandwidth term
    # scales with the per-merge row count = sum_b K_b (= B*K when all B
    # prompts share K). Split into a per-launch constant (CUDA launch +
    # warmup overhead — calibrated at K=8 small-rows) and a per-row slope
    # (memory-bound bandwidth, calibrated from the K=large measurement
    # minus the launch intercept). Both auto-fit by ``measure_merge_us``.
    #
    # Cascade depth-L (L ≥ 2) has (L-1) merges that run *sequentially* —
    # each merge depends on the prior level's output, so they can't be
    # parallelized across waves. Cost is therefore
    # ``(L-1) × (merge_launch_us + merge_bw_us_per_row × total_rows)``.
    # See _cascade_merge_us below.
    merge_launch_us: float = 2.0
    merge_bw_us_per_row: float = 0.0

    # Per-tile time table (µs per CTA-tile), nested by KV-cache dtype.
    # Outer key: canonical dtype label (DTYPE_FP16, DTYPE_BF16, ...).
    # Inner key: T_large ∈ T_LARGE_CHOICES + the small tile T_SMALL.
    # The value is the cost at L_kv = PER_TILE_L_KV_REF tokens; for
    # other L_kv the cost is computed by ``_tile_cost`` below using the
    # linear slope ``per_tile_per_kv_us`` (defaults to 0, i.e. constant
    # cost, for backward compatibility). Lookup via
    # ``lookup_per_tile_us(c, T, kv_dtype)`` which falls back to
    # ``DEFAULT_KV_DTYPE`` when the workload's dtype isn't calibrated.
    per_tile_us: dict[str, dict[int, float]] = field(default_factory=lambda: {
        DEFAULT_KV_DTYPE: {16: 0.6, 64: 1.4, 128: 2.4},
    })
    # Per-(KV token) slope of the prefill tile cost, fit from a two-point
    # probe at L_kv ∈ {PER_TILE_L_KV_REF, PER_TILE_L_KV_LONG}. Default 0
    # preserves the original constant-cost behavior; ``calibrate.py``
    # overwrites with the actual slope when the long-L_kv probe runs.
    # Nested by KV-cache dtype (see ``per_tile_us``).
    per_tile_per_kv_us: dict[str, dict[int, float]] = field(default_factory=lambda: {
        DEFAULT_KV_DTYPE: {16: 0.0, 64: 0.0, 128: 0.0},
    })
    # Per-call prefill kernel launch overhead. The fused-cascade wrapper
    # is ONE kernel launch regardless of cascade depth or wave count, so
    # this is added ONCE in the cost expression — not multiplied by wave
    # count. Fit from the multi-tile probe: at fixed L_kv, time vs wave
    # count is linear with intercept = launch and slope = per-wave cost.
    # Default 0 only takes effect when the slope is 0 (legacy caches),
    # in which case the cost expression falls back to ``waves ×
    # per_tile_us[T]`` with the launch baked into per_tile_us — see
    # ``_compute_us_from_tile_groups`` below.
    prefill_launch_us: float = 0.0
    # Per-wave additive overhead — paid for every wave even when the
    # wave's compute is small. Captures kernel-internal per-wave
    # scheduling cost that the slope×L_kv compute term misses. Fit from
    # the multi-tile probe: per_wave_overhead = (multi_tile_time -
    # single_tile_time) / (waves_2 - waves_1) - slope × L_kv. Indexed
    # by T (small variation across T expected).
    per_wave_overhead_us: dict[int, float] = field(default_factory=lambda: {
        16: 0.0,
        64: 0.0,
        128: 0.0,
    })

    # ------------------------------------------------------------------
    # SM count for wave-occupancy modeling. Tiles within a pool run in
    # parallel across SMs; the per-tile compute scales with
    # ceil(tile_count / num_sms), not tile_count. Calibrated from
    # ``torch.cuda.get_device_properties().multi_processor_count``.
    # A6000 = 84, RTX 6000 Ada = 142, H100 = 132.
    num_sms: int = 84

    # Decode-kernel per-(beam, kv-token) cost for the DEC_TAIL strategies.
    # Decode kernel is purpose-built for CTA_Q=1 and is bandwidth-bound
    # at our shapes — compute and bandwidth overlap rather than add, so
    # the cost expression uses max(bw_us, work_us) not bw_us + work_us.
    # Calibrated by `calibrate.measure_decode_per_kv_us`. Nested by KV
    # dtype (fp8 dequant overhead and _scaled_mm throughput differ from
    # fp16/bf16); lookup via ``lookup_decode_us_per_beam_kv_token``.
    decode_us_per_beam_kv_token: dict[str, float] = field(default_factory=lambda: {
        DEFAULT_KV_DTYPE: 0.0008,
    })
    # Per-step constant: post-launch / kernel-init cost for the decode
    # kernel. With the bs_kernel driver's plan_info cache the call is
    # essentially a kernel launch (~2 µs) per layer, no scheduling work.
    decode_launch_us: float = 2.0

    # Maximum cascade depth the picker enumerates as a candidate.
    # Default 3 reproduces legacy behavior (depth ∈ {2, 3}). Set higher
    # (typically 6) when the bs_kernel driver supports deeper cascades
    # AND the workloads have hierarchical sharing that depth>3 can
    # exploit. The picker still filters per-step against the workload's
    # actual intermediate-level count, so this is an upper bound, not
    # a forced depth.
    max_dispatch_depth: int = 3

    # FlashInfer prefill kernel's KV tile dimension (CTA_TILE_KV =
    # NUM_MMA_KV × NUM_WARPS_KV × 16). For CTA_TILE_Q ∈ {64, 128} with
    # NUM_WARPS_KV=1, dispatched values are {16, 32, 64, 128}. The cost
    # model uses this to count elemental (CTA_TILE_Q × CTA_TILE_KV ×
    # head_dim) sub-tiles inside each Q-tile: M_s = Σ tiles_q ×
    # ⌈L_kv / cta_tile_kv⌉. This is what KV split parallelizes across
    # SMs at runtime; counting sub-tiles before the ⌈·/N_SM⌉ ceiling
    # captures the parallelism the scheduler can extract.
    cta_tile_kv: int = 64
    # Decode kernel's KV chunk size — much smaller than prefill's
    # because the decode kernel is CTA_Q=1 and iterates KV in
    # vec_size-multiples (typically 8 tokens for fp16/head_dim=128/GQA).
    cta_tile_kv_decode: int = 8

    # Picker hysteresis margin against switching to a DEC_TAIL strategy.
    # ``pick_strategy_batch`` multiplies every DEC_TAIL-family candidate's
    # *comparison* cost by ``(1 + dec_tail_switch_margin)`` in the argmin, so
    # DEC_TAIL is selected only when it beats the best non-DEC_TAIL candidate
    # by at least this fraction. 0.0 = pure argmin (legacy behavior). A small
    # positive value (e.g. 0.10) delays the 1POOL→DEC_TAIL flip to a longer
    # tail: on long-decode self_consistency the picker flips ~2k steps before
    # the measured per-step crossover (DEC_TAIL's tail split-K is priced a bit
    # too cheap), and this margin shifts the switch back without re-deriving
    # the tail cost terms. Env-overridable via BS_KERNEL_DEC_TAIL_MARGIN at
    # coefficient load (see calibrate.load_or_defaults). Affects only the pick;
    # the raw cost is still reported in ``estimated_us`` / ``debug``.
    dec_tail_switch_margin: float = 0.0

    # Fused-tail inefficiency fraction (1POOL only). The fused-cascade 1POOL
    # kernel reads each beam's per-beam tail KV inside the same launch as the
    # shared-prefix MMA, so the tail can't get its own split-K grid and streams
    # at lower effective bandwidth than DEC_TAIL's dedicated CTA_Q=1 decode
    # kernel. Modeled as a penalty ``frac × (Σ tail tokens) × bytes_per_kv /
    # B_hbm`` added to the 1POOL cost. ~0 for short tails (the paper's picker
    # cells), so it doesn't perturb their picks; it grows with tail length and
    # is what makes the modeled 1POOL→DEC_TAIL crossover land near the measured
    # ~4k-token per-step crossover on long-decode self_consistency (instead of
    # relying on the discontinuous wave-quantization additive cliff). 0.0 =
    # legacy (no tail penalty). Fit offline against the long-decode crossover:
    # 0.07 lands the 1POOL→DEC_TAIL flip at ~4k tail tokens (matching the
    # measured per-step crossover) for the 1B K=16/B=1/L_p=32k cell, with no
    # regression on the 12-cell picker-demo oracle (short tails → penalty ≈ 0).
    fused_tail_ineff_frac: float = 0.07

    @classmethod
    def defaults(cls) -> "Coefficients":
        return cls()


def lookup_per_tile_us(c: Coefficients, T: int, kv_dtype: str) -> float:
    """Per-tile time at L_kv=PER_TILE_L_KV_REF for tile dim ``T`` and the
    given KV dtype. Falls back to ``DEFAULT_KV_DTYPE`` (then 0) when the
    workload's dtype isn't in the calibrated table.
    """
    table = c.per_tile_us.get(kv_dtype) or c.per_tile_us.get(DEFAULT_KV_DTYPE) or {}
    return table.get(T, 0.0)


def lookup_per_tile_per_kv_us(c: Coefficients, T: int, kv_dtype: str) -> float:
    """L_kv-slope of prefill tile cost for tile dim ``T`` and the given
    KV dtype. Falls back to ``DEFAULT_KV_DTYPE`` (then 0) when the
    workload's dtype isn't in the calibrated table.
    """
    table = c.per_tile_per_kv_us.get(kv_dtype) or c.per_tile_per_kv_us.get(DEFAULT_KV_DTYPE) or {}
    return table.get(T, 0.0)


def lookup_decode_us_per_beam_kv_token(c: Coefficients, kv_dtype: str) -> float:
    """Decode-kernel per-(beam, kv-token) cost for the given KV dtype.
    Falls back to ``DEFAULT_KV_DTYPE`` when the workload's dtype isn't
    in the calibrated table.
    """
    table = c.decode_us_per_beam_kv_token
    if kv_dtype in table:
        return table[kv_dtype]
    return table.get(DEFAULT_KV_DTYPE, 0.0008)


@dataclass
class WorkloadShape:
    """Per-step workload signature the cost model needs.

    ``L_p`` is the LCA prefix length in *tokens* (not pages). ``suffix_lens``
    is per-beam unique-tail length in tokens. ``intermediate`` describes
    a 3-level fork structure when one exists, else None.

    ``num_qo_heads`` is needed for the CTA-granularity wave model: the
    kernel packs ``K × gqa_group_size`` Q rows per beam-group, with
    ``gqa_group_size = num_qo_heads // num_kv_heads``. If unset (legacy
    workloads), defaults to ``num_kv_heads`` (gqa=1, no-op).
    """
    K: int
    L_p: int
    suffix_lens: list[int]
    num_kv_heads: int
    head_dim: int
    bytes_per_kv: int          # 2 (k+v) × num_kv_heads × head_dim × dtype_bytes
    intermediate: Optional["IntermediateShape"] = None
    num_qo_heads: int = 0      # 0 → fall back to num_kv_heads (gqa=1)
    # Canonical KV dtype label (DTYPE_FP16, DTYPE_BF16, DTYPE_FP8_E4M3,
    # ...). Selects which per-dtype rate the cost model uses for prefill
    # tile cost and decode-kernel cost. Defaults to fp16 so callers that
    # haven't been updated keep the legacy lookup behavior.
    kv_dtype: str = DEFAULT_KV_DTYPE

    def gqa_group_size(self) -> int:
        nq = self.num_qo_heads if self.num_qo_heads > 0 else self.num_kv_heads
        return max(1, nq // self.num_kv_heads)


@dataclass
class IntermediateShape:
    """Hierarchical-sharing structure for cascades of depth ≥ 3.

    ``levels`` is a list of intermediate levels; each level is a list of
    ``(beams_in_group, inter_len_tokens)`` tuples per group. ``levels[0]``
    is the level immediately below the shared root; ``levels[-1]`` is the
    level immediately above the per-beam tail. A depth-N cascade
    consumes the first (N-2) entries.

    Non-uniform group sizes and run lengths are allowed. Singleton
    groups (1 beam) contribute no bandwidth savings but are admitted so
    every query stays in a group at every level.

    For backward compatibility with the original 1-intermediate-level
    schema, ``IntermediateShape(groups=[...])`` and the ``.groups``
    property still work — they refer to the first (and only) intermediate
    level.
    """
    levels: list[list[tuple[int, int]]] = field(default_factory=list)

    @classmethod
    def from_groups(cls, groups: list[tuple[int, int]]) -> "IntermediateShape":
        """Construct from a single intermediate level (legacy schema)."""
        return cls(levels=[list(groups)])

    def __init__(
        self,
        levels: Optional[list[list[tuple[int, int]]]] = None,
        *,
        groups: Optional[list[tuple[int, int]]] = None,
    ):
        # Accept either ``levels`` (new) or ``groups`` (legacy) — never both.
        if levels is not None and groups is not None:
            raise ValueError("pass either 'levels' or 'groups', not both")
        if groups is not None:
            self.levels = [list(groups)]
        elif levels is not None:
            self.levels = [list(g) for g in levels]
        else:
            self.levels = []

    @property
    def groups(self) -> list[tuple[int, int]]:
        """First intermediate level (legacy accessor for depth-3 callers)."""
        return self.levels[0] if self.levels else []

    @property
    def max_depth(self) -> int:
        """Maximum cascade depth this shape supports (root + N inter + tail)."""
        return 2 + len(self.levels)


# ---------------------------------------------------------------------------
# Cost expressions
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# FlashInfer split-K heuristic constants (mirrors cascade.py:1080-1173).
# Below MIN_KV_LEN_FOR_SPLIT, the unsplit kernel runs fast enough that the
# extra +1 merge_states launch outweighs the parallelism gain.
_MIN_KV_LEN_FOR_SPLIT = 1024
_MIN_KV_CHUNK_TOKENS = 128
_SPLIT_K_PAGE_SIZE = 16  # mirrors page_size used in cascade.py heuristic


# L2 cache size used by the cross-level thrash term in cost_shared_batch.
# H100=50 MB, A100=40 MB, A6000/RTX-Ada≈40-100 MB. Conservative default
# = H100 size; future work: thread per-device value from calibrate.
_L2_SIZE_BYTES_DEFAULT = 50 * 1024 * 1024
# Empirical fit from benchmarks/bs_kernel/probe_c3_K_sweep.py at K ≥ 32
# on H100 + bf16 KV. Over-predicts at K ≤ 16 (where interference is
# small anyway). Captures the C3-style mispick mechanism (L1's K
# disjoint per-beam KV slabs evict L0's prefix lines when L0's working
# set exceeds L2).
_THRASH_ALPHA = 0.025


def _cross_level_thrash_us(
    workloads: list["WorkloadShape"], c: "Coefficients", depth: int,
) -> float:
    """Cross-level L2-thrash penalty for fused cascade (1POOL).

    Activates when L0's prefix working set exceeds L2 capacity AND
    there are ≥ 2 cascade levels in one kernel launch. L1's per-beam
    KV-slab accesses pollute L0's lines in L2, forcing L0 to refetch
    pages from HBM between scans.

    Formula (empirical, see project memory + probe_c3_K_sweep.py):
        interference = α × K × max(0, L_p × bpkv − L2_size) / B_hbm
    Activates per workload (so batched cells with B prompts sum the
    interference). Not applied for DEC_TAIL because it splits L0 and
    L1 into separate kernel launches → each kernel's working set is
    smaller and fits L2 (or at least doesn't get evicted by the
    other level mid-scan).
    """
    if depth < 2:
        return 0.0
    total = 0.0
    for w in workloads:
        L_p_bytes = w.L_p * w.bytes_per_kv
        excess = max(0, L_p_bytes - _L2_SIZE_BYTES_DEFAULT)
        if excess > 0:
            total += _THRASH_ALPHA * w.K * excess / c.B_hbm
    return total


# num_blocks_per_sm in FlashInfer's PrefillPlan (scheduler.cuh:717) — the
# grid budget is this × num_sms. Mirrored by the fused cascade's split-K.
_SPLIT_K_BLOCKS_PER_SM = 2


def _split_k_budget_tuples(num_kv_heads: int, num_sms: int) -> int:
    """Grid budget in (req, q_tile, kv_chunk) tuples: 2·num_sms / num_kv_heads."""
    return max((_SPLIT_K_BLOCKS_PER_SM * num_sms) // max(1, num_kv_heads), 1)


def _split_k_chunks_for_level(
    n_q_tiles_level: int,
    L_kv: int,
    *,
    num_kv_heads: int,
    num_sms: int,
    other_tiles: int = 0,
) -> int:
    """Per-level split-K decision, mirroring FlashInfer's PrefillPlan
    (scheduler.cuh:717) as ported into the fused cascade (cascade.py).
    Each (q_tile, kv_chunk) becomes one CTA, parallelized across SMs.

    Grid budget is ``2·num_sms / num_kv_heads`` tuples. ``other_tiles`` is
    the q_tile count contributed by the *other* levels of the same kernel
    launch — split this level only into the budget that's left after them
    (the launch-wide grid-saturation gate the picker was missing: a level
    that shares its launch with the per-beam tail (1POOL) gets no budget,
    while DEC_TAIL's prefix-only launch gets the full budget).
    """
    target_tuples = _split_k_budget_tuples(num_kv_heads, num_sms)
    budget = max(target_tuples - other_tiles, 0)
    if n_q_tiles_level >= budget:
        return 1
    target_chunks = max(budget // n_q_tiles_level, 1)
    chunk_size = max(_ceil_div(L_kv, target_chunks), _MIN_KV_CHUNK_TOKENS)
    chunk_size = (
        (chunk_size + _SPLIT_K_PAGE_SIZE - 1) // _SPLIT_K_PAGE_SIZE
    ) * _SPLIT_K_PAGE_SIZE
    return max(1, _ceil_div(L_kv, chunk_size))


def _bytes_to_us(n_bytes: int, c: Coefficients) -> float:
    return n_bytes / c.B_hbm


def _tile_compute_us(T: int, L_kv: int, c: Coefficients, kv_dtype: str) -> float:
    """Pure per-tile *compute* time (excludes one-shot kernel launch).

    With the L_kv slope calibrated, the per-tile cost is split into:
      * a constant launch overhead — paid ONCE per kernel call (added
        separately by ``_compute_us_from_tile_groups`` when the slope
        is set)
      * a per-tile compute term proportional to ``L_kv``

    The legacy ``per_tile_us[T]`` is the SUM of the two at L_kv_REF.
    Subtracting the launch intercept leaves the pure-compute term.

    ``kv_dtype`` selects the per-dtype calibration entry; falls back to
    the default dtype (fp16) when the workload's dtype isn't calibrated.
    """
    slope = lookup_per_tile_per_kv_us(c, T, kv_dtype)
    if slope == 0.0:
        # Legacy cache (no L_kv probe yet): per_tile_us is the full
        # constant-cost stand-in. Caller (_compute_us_from_tile_groups)
        # multiplies it by wave count and adds no separate launch — same
        # behavior as before the L_kv refactor.
        return lookup_per_tile_us(c, T, kv_dtype)
    # Pure compute = slope × L_kv (intercept is the launch overhead,
    # added once per call by the caller). Clamped to non-negative.
    return max(0.0, slope * L_kv)


def _tile_breakdown(
    levels: list[tuple[int, int, int, int]],
    *,
    pool_count: int,
    t_large: int,
) -> dict[tuple[int, int], int]:
    """Map ``(T, L_kv)`` → total tile count across the given cascade
    levels for a single prompt. Used by ``_compute_us_from_tile_groups``
    to aggregate across prompts and compute per-group wave time with
    L_kv-aware tile costs.
    """
    out: dict[tuple[int, int], int] = {}
    for g_count, beams_per_g, kv_tokens, _bpkv in levels:
        if kv_tokens == 0:
            continue
        if pool_count == 1:
            T = t_large
        else:
            T = T_SMALL if beams_per_g <= T_SMALL else t_large
        tiles_per_group = max(1, _ceil_div(beams_per_g, T))
        n = g_count * tiles_per_group
        key = (T, kv_tokens)
        out[key] = out.get(key, 0) + n
    return out


def _compute_us_from_tile_groups(
    groups: dict[tuple[int, int], int],
    c: Coefficients,
    kv_dtype: str,
) -> float:
    """Total compute time across (T, L_kv) tile groups for one kernel call.

    Two regimes:
      * **Slope set** (post-L_kv-calibration): cost =
        ``prefill_launch_us`` + Σ ``waves × slope[T] × L_kv``. The launch
        overhead is paid ONCE per call (the fused-cascade kernel is a
        single launch), and each (T, L_kv) tile group's pure compute
        scales with L_kv per token. This avoids the inflated B=1
        prediction the wave-multiplied form gave (where wrapper
        overhead bundled into per_tile_us was paid `waves` times).
      * **Slope all zero** (legacy cache): cost =
        Σ ``waves × per_tile_us[T]``. Matches the pre-refactor formula
        with launch baked into per_tile_us. No separate launch term.
    """
    has_slope = any(
        lookup_per_tile_per_kv_us(c, T, kv_dtype) > 0.0
        for (T, _) in groups.keys()
    )
    if not has_slope:
        total = 0.0
        for (T, L_kv), n_tiles in groups.items():
            if n_tiles <= 0:
                continue
            waves = max(1, _ceil_div(n_tiles, c.num_sms))
            total += waves * lookup_per_tile_us(c, T, kv_dtype)
        return total
    # Slope-based formula: launch once + per-wave (overhead + L_kv compute).
    # Per-wave time = max(per_wave_overhead[T], slope * L_kv) — the wave
    # can't run faster than its setup overhead, and at long L_kv the
    # compute dominates. Each wave still pays its setup even if its
    # tiles scan very short KV (the kernel-internal scheduling cost is
    # real — see picker B=8 mispick if this term is omitted).
    total = c.prefill_launch_us
    for (T, L_kv), n_tiles in groups.items():
        if n_tiles <= 0:
            continue
        waves = max(1, _ceil_div(n_tiles, c.num_sms))
        overhead = c.per_wave_overhead_us.get(T, 0.0)
        compute = _tile_compute_us(T, L_kv, c, kv_dtype)
        per_wave = max(overhead, compute)
        total += waves * per_wave
    return total


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


def _level_bw_us(
    levels: list[tuple[int, int, int, int]],
    c: Coefficients,
) -> float:
    """Total HBM-bandwidth time across cascade levels: sum(bytes) / B_hbm."""
    total_bytes = sum(
        g_count * kv_tokens * bytes_per_kv
        for g_count, beams_per_g, kv_tokens, bytes_per_kv in levels
        if kv_tokens > 0
    )
    return total_bytes / c.B_hbm


def _per_prompt_levels(w: WorkloadShape, depth: int) -> list[tuple[int, int, int, int]]:
    """Build the (g_count, beams_per_group, kv_tokens, bytes_per_kv) level
    tuples for one prompt's contribution to a cascade plan.

    Depth=2: emit (root, per-beam tail). All intermediate-level pages
    fold into the per-beam tail's bandwidth term (each beam reads them
    K times instead of being grouped).

    Depth=N (N ≥ 3): emit (root, intermediate_0, ..., intermediate_{N-3},
    per-beam tail). Uses the first (N-2) entries of
    ``w.intermediate.levels``. Any deeper intermediate levels in the
    workload that aren't consumed at this depth fold back into the
    per-beam tail.

    Caller is responsible for ensuring ``w.intermediate`` has at least
    ``N-2`` levels when ``depth=N`` is requested.

    ``suffix_lens`` in the workload reflects the layout produced with
    the deepest available intermediate decomposition — i.e., per-beam
    tails *after* the deepest intermediate run.
    """
    levels: list[tuple[int, int, int, int]] = []
    # Level 0: 1 group of K beams sharing L_p tokens.
    levels.append((1, w.K, w.L_p, w.bytes_per_kv))
    avg_suffix_post = sum(w.suffix_lens) / max(1, len(w.suffix_lens))

    inter_levels = w.intermediate.levels if w.intermediate is not None else []
    n_inter_used = depth - 2
    assert 0 <= n_inter_used <= len(inter_levels), (
        f"depth={depth} requires {n_inter_used} intermediate levels, "
        f"workload has {len(inter_levels)}"
    )

    # Used intermediate levels: emit one tuple per group at each used level.
    for level_groups in inter_levels[:n_inter_used]:
        for beams_in_grp, inter_len in level_groups:
            levels.append((1, beams_in_grp, inter_len, w.bytes_per_kv))

    # Unused intermediate levels (those past the cascade's depth) fold
    # back into the per-beam tail. Each beam reads the unused-level pages
    # independently — the per-beam-average extra tokens equal
    # `sum_over_unused_levels(sum(beams * inter_len for groups)) / K`.
    extra_per_beam = 0.0
    for level_groups in inter_levels[n_inter_used:]:
        extra_per_beam += sum(g[0] * g[1] for g in level_groups) / max(1, w.K)
    avg_suffix = avg_suffix_post + extra_per_beam

    # Last level: K groups of 1 beam each.
    levels.append((w.K, 1, int(round(avg_suffix)), w.bytes_per_kv))
    return levels


def _cascade_merge_us(
    depth: int,
    workloads: list[WorkloadShape],
    c: Coefficients,
) -> float:
    """Cost of the (depth - 1) ``merge_state_in_place`` launches that
    sit between cascade levels.

    Both FUSED and DEC_TAIL cascades pay this — the fused-cascade
    wrapper collapses prefill into one launch but leaves each merge as
    a separate launch (see ``FusedMultiLevelCascadeAttentionWrapper``).
    Merges are sequential (each depends on the prior level's output) so
    cost is ``(depth - 1) × per_merge_us``, no wave division.

    Per-merge cost is launch overhead + bandwidth. The bandwidth term
    scales with the total query row count across the batch
    (= sum_b K_b), because the kernel reads two
    ``(rows, num_qo_heads, head_dim)`` tensors plus their softmax-state
    counterparts and writes one of each. The per-row µs constant
    ``merge_bw_us_per_row`` is calibrated against the specific
    deployment's ``num_qo_heads × head_dim`` (see
    ``calibrate.measure_merge_us``).
    """
    n_merges = max(0, depth - 1)
    if n_merges == 0:
        return 0.0
    total_rows = sum(w.K for w in workloads)
    per_merge = c.merge_launch_us + c.merge_bw_us_per_row * total_rows
    return n_merges * per_merge


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
    trace: Optional[dict] = None,
) -> float:
    """Cost of the shared-prefix path running B prompts in one cascade
    launch.

    All prompts use the same ``(depth, pool_count, t_large)`` — that's
    the choice the picker has to make for the fused kernel. Within
    each prompt, the per-level tile counts are the same as the B=1
    case. Across prompts, tile counts SUM, then wave count is computed
    against ``num_sms`` ONCE across the batch — this is the cross-prompt
    wave-occupancy effect that B=1 reasoning misses.

    depth ≥ 2. For depth=N (N ≥ 3), every prompt's workload must have
    at least N-2 intermediate levels (else the picker should skip this
    candidate at the batch — the assertion in ``_per_prompt_levels``
    surfaces a programming error rather than a workload mismatch).

    ``fused_merge`` is retained as an API-stable no-op kwarg; merge cost
    is no longer modeled (see module docstring).
    """
    del fused_merge  # no longer modeled
    assert depth >= 2, depth
    assert pool_count in (1, 2), pool_count
    assert t_large in T_LARGE_CHOICES, t_large
    if depth >= 3:
        n_inter_needed = depth - 2
        for w in workloads:
            n_have = (
                len(w.intermediate.levels) if w.intermediate is not None else 0
            )
            assert n_have >= n_inter_needed, (
                f"depth={depth} batch needs {n_inter_needed} intermediate "
                f"levels on every prompt; one prompt has {n_have}"
            )

    # CTA-granularity wave model (replaces prior elemental-sub-tile model).
    #
    # The fused-cascade kernel schedules CTAs onto SMs, not elemental
    # (T_Q × T_KV) sub-tiles. Per physical level:
    #   * packed Q rows = beams_per_g · gqa_group_size  (kernel-true Q packing)
    #   * q_tiles       = g_count · ⌈packed/T⌉
    #   * num_chunks    = FlashInfer split-K decision per level (cascade.py)
    #   * CTAs          = q_tiles · num_chunks · num_kv_heads  (gridDim.z = kv_heads)
    #   * chunk_L_kv    = ⌈L_kv/num_chunks⌉
    #   * τ_CTA         = ⌈chunk_L_kv/cta_tile_kv⌉ · τ_elem
    #
    # Within one CTA_TILE_Q pool the kernel launches all CTAs together →
    # ⌈Σ CTAs / N_SM⌉ waves, each wave's duration dominated by the
    # heaviest CTA in the pool (max τ_CTA across levels routed to that
    # pool). Separate pools = separate kernel launches → independent ceil
    # per pool, summed compute.
    #
    # This captures the wave-quantization knee verified by
    # ``benchmarks/bs_kernel/probe_cascade_thin_q.py``: adding a thin-Q
    # cascade level whose CTAs tip total SM occupancy over a wave
    # boundary adds ~max_τ_CTA (heavy L0 cost), not its own marginal
    # compute — exactly what the old elemental-sub-tile model missed.
    # All workloads in a batch share KV dtype; pick from the first.
    kv_dtype = workloads[0].kv_dtype if workloads else DEFAULT_KV_DTYPE
    slope = lookup_per_tile_per_kv_us(c, t_large, kv_dtype)
    if slope > 0:
        tau_elem = slope * c.cta_tile_kv
    else:
        tau_elem = lookup_per_tile_us(c, t_large, kv_dtype) * c.cta_tile_kv / max(1, PER_TILE_L_KV_REF)

    # First pass: aggregate per-level q_tile counts ACROSS the batch.
    # FlashInfer's split-K decision is per-level over all B prompts in
    # the cascade (not per-workload), so the cost model must match: sum
    # n_q_tiles across all workloads at each level *before* picking
    # num_chunks. Otherwise at B > 1 the model over-splits L0 (each
    # workload thinks it has too few q_tiles → aggressive split-K)
    # which inflates L0's CTA count and shrinks τ_L0 — wrong direction
    # for the picker on batched cells.
    #
    # Track per-level: total n_q_tiles across batch, L_kv (uniform per
    # level since all prompts share dtype/shape), and a representative
    # num_kv_heads (uniform across workloads).
    level_aggregates: dict[tuple[int, int], dict] = {}  # (T, L_kv) → state
    bw_us = 0.0
    for w in workloads:
        levels = _per_prompt_levels(w, depth)
        gqa = w.gqa_group_size()
        for g_count, beams_per_g, L_kv, _bpkv in levels:
            if L_kv == 0:
                continue
            packed_phys = beams_per_g * gqa
            if pool_count == 1:
                T = t_large
            else:
                T = T_SMALL if packed_phys <= T_SMALL else t_large
            n_q_tiles = g_count * max(1, _ceil_div(packed_phys, T))
            key = (T, L_kv)
            state = level_aggregates.setdefault(key, {
                "n_q_tiles": 0, "num_kv_heads": w.num_kv_heads,
            })
            state["n_q_tiles"] += n_q_tiles
        bw_us += _level_bw_us(levels, c)

    # Now compute num_chunks once per (pool, L_kv) group using the
    # batch-aggregated n_q_tiles, then build per-pool entry list.
    pool_max_tau_cta: dict[int, float] = {}
    pool_level_entries: dict[int, list[tuple[int, float]]] = {}
    # Launch-wide q_tile total: all levels of this cascade run in ONE kernel,
    # so split-K of any one level only gets the grid budget left after the
    # others (FlashInfer's grid-saturation gate). For 1POOL the per-beam tail
    # level's tiles fill the grid → L0 can't split (matches the kernel); for
    # DEC_TAIL the tail is a separate launch (not in level_aggregates here).
    total_q_tiles = sum(st["n_q_tiles"] for st in level_aggregates.values())
    for (T, L_kv), state in level_aggregates.items():
        n_q_tiles = state["n_q_tiles"]
        num_kv_heads = state["num_kv_heads"]
        num_chunks = _split_k_chunks_for_level(
            n_q_tiles, L_kv,
            num_kv_heads=num_kv_heads, num_sms=c.num_sms,
            other_tiles=total_q_tiles - n_q_tiles,
        )
        chunk_L_kv = _ceil_div(L_kv, num_chunks)
        n_ctas = n_q_tiles * num_chunks * num_kv_heads
        kv_iters_per_cta = max(1, _ceil_div(chunk_L_kv, c.cta_tile_kv))
        tau_cta = kv_iters_per_cta * tau_elem
        pool_level_entries.setdefault(T, []).append((n_ctas, tau_cta))
        if tau_cta > pool_max_tau_cta.get(T, 0.0):
            pool_max_tau_cta[T] = tau_cta

    # Equivalent-CTA wave model: convert light CTAs to "heavy-equivalent
    # units" by τ ratio, then wave-quantize the equivalents. This models
    # the GPU scheduler's work-stealing: a light CTA on an SM that just
    # finished a heavy CTA finishes in (τ_light / τ_heavy) fraction of
    # the wave time, so it consumes only that fraction of one SM slot.
    #
    # Rationale: the previous "⌈total_CTAs / num_sms⌉ · max_τ" formula
    # over-predicts at moderate batches (B=4-8 with K=16-32) where L1's
    # many light CTAs inflate the wave count but the kernel actually
    # absorbs them into L0's wave via work-stealing.
    #
    # The "just barely tips" boundary still triggers when L0's CTAs
    # alone are near num_sms and L1 has any non-trivial heavy-equivalent
    # weight (e.g., 70B C3: L0=128, L1=4.6 equiv → 132.6 > 132 → 2 waves).
    # Picker demo: 7/12 (vs 6/12 for ⌈total/SM⌉·max_τ).
    #
    # Tried "spare-capacity absorption" (light work absorbed in heavy
    # wave's spare SM-time, overflow at avg rate): fixed A3 but broke
    # C3 and B3. The two models capture different physical effects and
    # neither dominates — equivalent-CTA wins because it preserves the
    # user-flagged C3 mispick.
    compute_us = 0.0
    total_waves = 0
    for T, entries in pool_level_entries.items():
        max_tau = pool_max_tau_cta.get(T, 0.0)
        if max_tau <= 0.0:
            continue
        equiv = 0.0
        for n_ctas, tau_cta in entries:
            equiv += n_ctas * (tau_cta / max_tau)
        waves = max(1, _ceil_div(int(math.ceil(equiv)), c.num_sms))
        total_waves += waves
        compute_wave_us = waves * max_tau

        # Light-level overflow penalty (the C6 high-K + short-L_p case).
        #
        # Equivalent-CTA accounting compresses light CTAs into fractional
        # heavy-equivalent units, assuming they hide in the heavy level's
        # wave via work-stealing. When the heavy-τ level (typically L0)
        # has LESS total work than the lighter level(s) (typically L1 with
        # many thin Q-tiles in batched workloads), the light excess can't
        # hide — there isn't enough heavy work to overlap with. Pay the
        # excess at balanced wall rate.
        #
        # Fires at e.g. C6 (70B mcr K=64 B=4 L_p=500): L0 work=471 SM-µs,
        # L1 work=1884 → overflow 1413 → +10.7 µs forces correct DT pick.
        # Doesn't fire when L0 dominates (the typical regime: long prefix
        # with few beams), so no regression on the 1POOL-favoring cells.
        # See benchmarks/bs_kernel/verify_picker_demo_oracle.py.
        heavy_work_us = 0.0
        light_work_us = 0.0
        for n_ctas, tau_cta in entries:
            work = n_ctas * tau_cta
            if abs(tau_cta - max_tau) < 1e-9:
                heavy_work_us += work
            else:
                light_work_us += work
        light_overflow_us = max(0.0, light_work_us - heavy_work_us) / max(1, c.num_sms)

        compute_us += compute_wave_us + light_overflow_us

    merge_rounds = int(math.ceil(math.log2(max(2, depth))))
    merge_us = merge_rounds * c.merge_launch_us

    # Cross-level L2 cache-thrash penalty (the C3-mispick mechanism).
    # See _cross_level_thrash_us docstring + project memory
    # [[project_cascade_l2_thrash]] for the L_p × K × (excess over L2)
    # empirical formula.
    thrash_us = _cross_level_thrash_us(workloads, c, depth)

    # Wave-count-and-batch-gated bw/compute overlap.
    #
    # FA-style kernels pipeline HBM with MMA so that wall ≈ max(bw,
    # compute) only when:
    #   (a) the kernel runs in ≤ 2 waves (so compute can hide under bw
    #       within a wave), AND
    #   (b) per-wave HBM demand fits the bandwidth budget — which
    #       roughly tracks B. At high B each wave's KV reads multiply
    #       and the controller saturates regardless of wave count.
    # Either condition failing pushes the kernel toward additive
    # bw + compute. ``total_waves`` is the sum of per-pool wave counts
    # under the equivalent-CTA model.
    B = len(workloads)
    # Per-wave HBM demand tracks the total beam count (B·K), not just B:
    # the fused 1POOL kernel reads every beam's per-beam tail KV each wave,
    # so at high B·K (e.g. F2 multi_few_shot K=128 B=4 → 512 beams) the
    # memory controller saturates and the tail bandwidth no longer hides
    # under prefix compute — push to additive bw+compute. (DEC_TAIL splits
    # the tail into a separate decode launch and already pays it additively,
    # so this restores the correct DEC_TAIL<1POOL ordering there. F4
    # self_consistency K=16 B=1 = 16 beams stays well-pipelined → 1POOL.)
    # Overlap gate: the additive (bw+compute) form applies when the memory
    # controller saturates — driven by beam concurrency (B·K), NOT raw wave
    # count. The old ``total_waves <= 2`` term created a discontinuous cost
    # cliff when a long per-beam tail tipped a LOW-beam cell from 2→3 waves
    # (the F4 self_consistency early-flip artifact): low B·K never saturates
    # the controller regardless of wave count, so it stays pipelined. The
    # smooth tail penalty below supplies the 1POOL→DEC_TAIL crossover that the
    # wave cliff used to provide crudely. High B·K (e.g. F2 K=128 B=4 = 512
    # beams) still goes additive via the n_beams guard.
    n_beams = sum(w.K for w in workloads)
    well_pipelined = (B <= 8) and (n_beams <= 128)
    work_us = (
        max(bw_us, compute_us + thrash_us)
        if well_pipelined
        else bw_us + compute_us + thrash_us
    )

    # Fused-tail inefficiency penalty (1POOL only): the per-beam tail KV is
    # streamed inside the fused launch at lower effective bandwidth than
    # DEC_TAIL's dedicated decode kernel (no own split-K). ~0 for short tails,
    # grows linearly with tail → supplies the smooth long-tail crossover. See
    # Coefficients.fused_tail_ineff_frac.
    tail_penalty_us = 0.0
    if c.fused_tail_ineff_frac:
        tail_bytes = sum(sum(w.suffix_lens) * w.bytes_per_kv for w in workloads)
        tail_penalty_us = c.fused_tail_ineff_frac * tail_bytes / c.B_hbm
        work_us += tail_penalty_us

    if trace is not None:
        trace.update({
            "bw_us": bw_us, "compute_us": compute_us, "thrash_us": thrash_us,
            "merge_us": merge_us, "total_waves": total_waves,
            "n_beams": n_beams, "B": B, "well_pipelined": well_pipelined,
            "tail_penalty_us": tail_penalty_us,
            "work_us": work_us, "total": work_us + merge_us,
        })

    return work_us + merge_us


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
    inter_levels = w.intermediate.levels if w.intermediate is not None else []
    n_inter_used = depth - 2
    assert 0 <= n_inter_used <= len(inter_levels), (
        f"depth={depth} requires {n_inter_used} intermediate levels, "
        f"workload has {len(inter_levels)}"
    )
    for level_groups in inter_levels[:n_inter_used]:
        for beams_in_grp, inter_len in level_groups:
            levels.append((1, beams_in_grp, inter_len, w.bytes_per_kv))
    return levels


def cost_decode_tail_batch(
    workloads: list[WorkloadShape], c: Coefficients,
) -> float:
    """Tail term for the DEC_TAIL strategies: decode_launch + bw + compute.

    CTA-granularity wave model (mirroring ``cost_shared_batch``):
      * Each beam is one CTA (CTA_Q=1) in the decode kernel.
      * When ``n_beams < num_sms``, FlashInfer's decode kernel does
        internal split-KV: each beam's KV is split into ``split_kv``
        chunks, spawning ``n_beams · split_kv`` total CTAs to fill SMs.
        We mirror the heuristic ``split_kv ≈ num_sms / n_beams``,
        clamped so each chunk holds ≥ ``cta_tile_kv_decode`` tokens.
      * Per-CTA τ = ⌈(mean_tail / split_kv) / cta_tile_kv_decode⌉ ·
        τ_tail_elem (each CTA iterates its KV slice sequentially in
        elemental tiles, like ``cost_shared_batch`` does for prefill).
      * Compute = ⌈total_CTAs / num_sms⌉ · τ_CTA — one wave when
        split-KV saturates SMs, multiple waves at large K when no
        split-KV is needed.

    Bandwidth and compute are summed (not max'd) — the previous max
    form assumed perfect overlap inside the decode kernel; the additive
    form is the no-overlap worst case and tends to penalize DEC_TAIL
    when its tail-side compute is non-trivial. (Tested swapping to max
    on the picker_demo grid; at our shapes work_us ≈ 0.05 µs ≪ bw_us,
    so max ≈ bw_us and the change moved zero picks. Kept the additive
    form documented above.)

    The extra ``decode_launch_us`` term accounts for the extra kernel
    launch DEC_TAIL pays over FUSED. Without this term the picker
    misses the FUSED-vs-DECTAIL crossover at small-B / long-L_p where
    both strategies are bandwidth-bound on the prefix and the extra
    launch is the only distinguishing cost.

    TODO: when split_kv > 1 the decode kernel also fires a
    ``merge_states`` launch to fold the split-KV partials per beam.
    That's a further ~merge_launch_us currently not modeled.
    """
    total_kv_tokens = 0
    n_beams = 0
    bytes_per_kv = 0
    for w in workloads:
        total_kv_tokens += sum(w.suffix_lens)
        n_beams += w.K
        bytes_per_kv = w.bytes_per_kv  # all workloads share dtype/shape
    if n_beams == 0:
        return 0.0
    mean_tail = total_kv_tokens / n_beams
    bw_us = total_kv_tokens * bytes_per_kv / c.B_hbm

    # Per-dtype decode rate (fp8 dequant cost differs from fp16/bf16).
    kv_dtype = workloads[0].kv_dtype if workloads else DEFAULT_KV_DTYPE
    decode_per_token = lookup_decode_us_per_beam_kv_token(c, kv_dtype)

    # Split-KV factor: how many CTAs per beam does the decode kernel
    # spawn to fill SMs? When K saturates SMs no split is needed; else
    # split each beam's KV until total CTAs ≈ num_sms, never below one
    # elemental tile per chunk.
    mean_tail_int = max(1, int(round(mean_tail)))
    max_split_by_kv = max(1, mean_tail_int // c.cta_tile_kv_decode)
    split_kv = max(1, c.num_sms // max(1, n_beams))
    split_kv = min(split_kv, max_split_by_kv)

    n_ctas = n_beams * split_kv
    kv_iters_per_cta = max(
        1, _ceil_div(_ceil_div(mean_tail_int, split_kv), c.cta_tile_kv_decode),
    )
    tau_tail_elem = decode_per_token * c.cta_tile_kv_decode
    tau_cta = kv_iters_per_cta * tau_tail_elem
    waves = max(1, _ceil_div(n_ctas, c.num_sms))
    work_us = waves * tau_cta
    # Within the decode kernel, HBM load and MMA pipeline overlap →
    # max(bw, work) is more accurate than sum (matches the form used
    # by cost_shared_batch and cost_dec_tail_batch's prefix term).
    return c.decode_launch_us + max(bw_us, work_us)


def cost_dec_tail_batch(
    workloads: list[WorkloadShape],
    c: Coefficients,
    *,
    depth: int,
    t_large: int = 64,
) -> float:
    """Hybrid: prefix (and optional intermediate) via prefill cascade +
    per-beam tail via paged decode + 1 (or 2) merges.

    All prompts share one ``(depth, t_large)``. ``t_large`` is the
    CTA_TILE_Q the kernel will use for the prefix-side fused cascade
    (FlashInfer's adaptive router picks it from the prefix's packed
    Q size; the picker enumerates both ``T_LARGE_CHOICES`` so it can
    mirror that choice — see ``pick_strategy_batch``).

    Previous hardcode ``t_large=64`` over-priced the prefix at high-GQA
    workloads (e.g. 70B with packed_qo=K·8=256 → kernel routes to T=128,
    but model priced at T=64 → kv_iters per CTA doubles, compute
    doubles). This biased the picker against DEC_TAIL on 70B C3 cells
    where DT is empirically faster.

    depth=N (N ≥ 3) requires every prompt's workload to have at least
    N-2 intermediate levels; the picker filters that.
    """
    assert depth >= 2, depth
    assert t_large in T_LARGE_CHOICES, t_large
    if depth >= 3:
        n_inter_needed = depth - 2
        for w in workloads:
            n_have = (
                len(w.intermediate.levels) if w.intermediate is not None else 0
            )
            assert n_have >= n_inter_needed, (
                f"depth={depth} dec_tail needs {n_inter_needed} intermediate "
                f"levels on every prompt; one prompt has {n_have}"
            )

    # CTA-granularity wave model for the prefix-side fused-cascade launch.
    # See ``cost_shared_batch`` for rationale and the formula breakdown.
    t_large_prefix = t_large
    kv_dtype = workloads[0].kv_dtype if workloads else DEFAULT_KV_DTYPE
    slope = lookup_per_tile_per_kv_us(c, t_large_prefix, kv_dtype)
    if slope > 0:
        tau_elem = slope * c.cta_tile_kv
    else:
        tau_elem = lookup_per_tile_us(c, t_large_prefix, kv_dtype) * c.cta_tile_kv / max(1, PER_TILE_L_KV_REF)

    pool_ctas: dict[int, int] = {}
    pool_max_tau_cta: dict[int, float] = {}
    bw_us = 0.0
    for w in workloads:
        levels = _per_prompt_levels_no_tail(w, depth)
        gqa = w.gqa_group_size()
        for g_count, beams_per_g, L_kv, _bpkv in levels:
            if L_kv == 0:
                continue
            packed_phys = beams_per_g * gqa
            T = t_large_prefix  # pool_count=1 here
            n_q_tiles = g_count * max(1, _ceil_div(packed_phys, T))
            num_chunks = _split_k_chunks_for_level(
                n_q_tiles, L_kv,
                num_kv_heads=w.num_kv_heads, num_sms=c.num_sms,
            )
            chunk_L_kv = _ceil_div(L_kv, num_chunks)
            n_ctas = n_q_tiles * num_chunks * w.num_kv_heads
            kv_iters_per_cta = max(1, _ceil_div(chunk_L_kv, c.cta_tile_kv))
            tau_cta = kv_iters_per_cta * tau_elem
            pool_ctas[T] = pool_ctas.get(T, 0) + n_ctas
            if tau_cta > pool_max_tau_cta.get(T, 0.0):
                pool_max_tau_cta[T] = tau_cta
        bw_us += _level_bw_us(levels, c)

    prefix_compute_us = 0.0
    for T, n_ctas in pool_ctas.items():
        waves = max(1, _ceil_div(n_ctas, c.num_sms))
        prefix_compute_us += waves * pool_max_tau_cta.get(T, 0.0)

    merge_rounds = int(math.ceil(math.log2(max(2, depth))))
    merge_us = merge_rounds * c.merge_launch_us
    # Prefix kernel: max(bw, compute) within the launch (FA pipelines
    # HBM with MMA). See cost_shared_batch for rationale.
    prefix_us = max(bw_us, prefix_compute_us) + merge_us

    # Tail under decode kernel — max(bw, wave-quantized elemental compute).
    tail_us = cost_decode_tail_batch(workloads, c)

    return prefix_us + tail_us


def cost_xprompt_dec_tail_batch(
    workloads: list[WorkloadShape],
    c: Coefficients,
    *,
    cross_prompt_lca: int,
    t_large: int = 64,
) -> float:
    """Cross-prompt 3-level dec_tail cost.

    Same hybrid shape as ``cost_dec_tail_batch`` (prefill cascade + per-beam
    decode tail + merges), but the prefill levels are built CROSS-PROMPT
    rather than per-prompt:

      * L0 = ONE group of all ``B*K`` beams reading the shared sys
        (``cross_prompt_lca`` tokens) — counted **once** in the bandwidth
        term (``_level_bw_us`` prices ``g_count * kv_tokens``, independent of
        the beams attending it), vs ``B``× when each per-prompt group re-lists
        sys in the SHARED_* strategies.
      * L1 = one group per prompt of ``K`` beams reading that prompt's private
        bundle (``L_p - cross_prompt_lca`` tokens) — distinct per prompt, so no
        dedup; identical bytes to the per-prompt path.
      * tail = per-beam decode kernel — identical to the SHARED_*_DEC_TAIL tail
        (``cost_decode_tail_batch``).

    The single B*K L0 group flows through the SAME tile/wave/bandwidth
    primitives as ``cost_dec_tail_batch`` (``_level_tiles_and_bytes`` logic
    inlined here, ``_split_k_chunks_for_level``, ``_level_bw_us``,
    ``_cross_level_thrash_us``), so the read saving (sys 1× vs B×) and the
    occupancy of collapsing B*K into one query-tile pool are both counted
    honestly — the short-sys-loses / long-sys-wins crossover emerges from
    ``max(bw, compute)`` rather than a hand-tuned credit.

    Returns ``math.inf`` when not applicable (``B < 2`` or no shared sys), so
    the picker never selects it in those cases.
    """
    assert t_large in T_LARGE_CHOICES, t_large
    B = len(workloads)
    if B < 2 or cross_prompt_lca <= 0:
        return math.inf
    w0 = workloads[0]
    bpkv = w0.bytes_per_kv
    gqa = w0.gqa_group_size()
    num_kv_heads = w0.num_kv_heads
    n_beams_total = sum(w.K for w in workloads)

    # Cross-prompt prefill levels (no tail): L0 sys read once, L1 per-prompt
    # bundles. Each entry is (group_count, beams_per_group, kv_tokens, bpkv).
    levels: list[tuple[int, int, int, int]] = [
        (1, n_beams_total, cross_prompt_lca, bpkv),
    ]
    for w in workloads:
        bundle = w.L_p - cross_prompt_lca
        if bundle > 0:
            levels.append((1, w.K, bundle, bpkv))

    kv_dtype = w0.kv_dtype
    slope = lookup_per_tile_per_kv_us(c, t_large, kv_dtype)
    if slope > 0:
        tau_elem = slope * c.cta_tile_kv
    else:
        tau_elem = (
            lookup_per_tile_us(c, t_large, kv_dtype)
            * c.cta_tile_kv / max(1, PER_TILE_L_KV_REF)
        )

    pool_ctas = 0
    pool_max_tau_cta = 0.0
    for g_count, beams_per_g, L_kv, _bpkv in levels:
        if L_kv == 0:
            continue
        packed_phys = beams_per_g * gqa
        n_q_tiles = g_count * max(1, _ceil_div(packed_phys, t_large))
        num_chunks = _split_k_chunks_for_level(
            n_q_tiles, L_kv, num_kv_heads=num_kv_heads, num_sms=c.num_sms,
        )
        chunk_L_kv = _ceil_div(L_kv, num_chunks)
        n_ctas = n_q_tiles * num_chunks * num_kv_heads
        kv_iters_per_cta = max(1, _ceil_div(chunk_L_kv, c.cta_tile_kv))
        tau_cta = kv_iters_per_cta * tau_elem
        pool_ctas += n_ctas
        if tau_cta > pool_max_tau_cta:
            pool_max_tau_cta = tau_cta

    waves = max(1, _ceil_div(pool_ctas, c.num_sms))
    prefix_compute_us = waves * pool_max_tau_cta

    # L2-aware bandwidth — the crux of when cross-prompt actually wins.
    # The cross-prompt L0 physically reads the shared sys ONCE; the per-prompt
    # baseline (cost_dec_tail_batch) re-reads it once per group. But those
    # re-reads HIT L2 (free) unless one group's private bundle evicts sys
    # before the next group re-reads it (LRU reuse distance ≈ sys + bundle).
    # So the sys-read saving is real only ABOVE L2: below it, both strategies
    # effectively read sys ~once, so we charge cross-prompt as if it also paid
    # the (L2-served) re-reads — it ties the per-prompt baseline there and wins
    # only once per-layer sys+bundle exceeds L2. ``evict`` is the fraction of
    # sys spilled from L2 per group.
    sys_bytes = cross_prompt_lca * bpkv
    max_bundle_tok = max((w.L_p - cross_prompt_lca) for w in workloads)
    bundle_resident_bytes = max(0, max_bundle_tok) * bpkv
    evict = min(1.0, max(0.0,
        (sys_bytes + bundle_resident_bytes - _L2_SIZE_BYTES_DEFAULT)
        / max(1, sys_bytes)))
    sys_eff_bytes = sys_bytes * (1.0 + (B - 1) * (1.0 - evict))
    bundle_total_bytes = sum(
        max(0, w.L_p - cross_prompt_lca) for w in workloads) * bpkv
    bw_us = _bytes_to_us(int(sys_eff_bytes + bundle_total_bytes), c)

    # 2 prefill cascade levels (sys, bundle) + decode-tail merge → like a
    # depth-3 dec_tail.
    merge_rounds = int(math.ceil(math.log2(3)))
    merge_us = merge_rounds * c.merge_launch_us
    prefix_us = max(bw_us, prefix_compute_us) + merge_us
    tail_us = cost_decode_tail_batch(workloads, c)
    return prefix_us + tail_us


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
    cross_prompt_lca: int = 0,
) -> Pick:
    """Pick the best strategy for a *batch* of prompts decoded together.

    All prompts share one ``(depth, pool_count, t_large)`` — that's the
    choice forced by a single fused kernel launch. Tile counts sum
    across prompts, so cross-prompt wave overflow tilts the picker
    toward the 2-pool path at large B (where the per-beam tile count
    crosses ``num_sms`` and a separate T_SMALL pool reduces wave count).

    A depth-N candidate is only considered if every prompt's workload
    has at least N-2 intermediate levels (otherwise the cascade plan
    can't use an N-level layout consistently — flashinfer wraps a
    single ``num_levels`` per wrapper instance).

    When ``available_strategies is None`` the picker defaults to
    ``DEFAULT_STRATEGIES`` (PER_BEAM, SHARED_2L_1POOL, SHARED_3L_1POOL,
    SHARED_2L_DEC_TAIL). Pass an explicit set to widen / override.
    """
    if available_strategies is None:
        available_strategies = DEFAULT_STRATEGIES
    debug: dict[str, float] = {}

    # Per-beam baseline.
    pb_cost = cost_per_beam_batch(workloads, c)
    debug["per_beam"] = pb_cost
    candidates: list[tuple[Strategy, float, int, int, int]] = [
        (Strategy.PER_BEAM, pb_cost, 0, 1, 1),
    ]

    # Per-batch maximum supported depth: min over workloads' intermediate
    # depths (depth=2 always supported, depth=N requires N-2 intermediate
    # levels on every prompt).
    def _supported_depth(w: WorkloadShape) -> int:
        if w.intermediate is None:
            return 2
        return 2 + len(w.intermediate.levels)

    batch_max_depth = min(
        (_supported_depth(w) for w in workloads), default=2
    )
    enum_max_depth = min(c.max_dispatch_depth, batch_max_depth)

    # Prefill cascade candidates. Skip whole (depth, pool_count) combos
    # whose Strategy isn't in ``available_strategies`` — avoids the
    # ``cost_shared_batch`` call entirely (was ~5-10 µs each × 20 combos
    # = significant for tiny workloads).
    for depth in range(2, enum_max_depth + 1):
        for pool_count in (1, 2):
            s = _strategy_for(depth, "shared", pool_count)
            if s not in available_strategies:
                continue
            for t_large in T_LARGE_CHOICES:
                cs = cost_shared_batch(
                    workloads, c,
                    depth=depth, pool_count=pool_count,
                    t_large=t_large, fused_merge=fused_merge,
                )
                tag = f"shared_d{depth}_p{pool_count}_t{t_large}"
                debug[tag] = cs
                candidates.append((s, cs, t_large, depth, pool_count))

    # DEC_TAIL candidates: prefix(/intermediates) prefill + decode tail +
    # merge. T_large applies to the prefix-side fused cascade (the tail's
    # decode kernel is CTA_Q=1, T_large doesn't apply there). Enumerate
    # both T_LARGE_CHOICES so the picker can mirror what FlashInfer's
    # adaptive router would pick (T=128 for high-GQA / large packed Q,
    # T=64 for narrower Q). Previously hardcoded to T=64, which inflated
    # prefix compute at 70B-class GQA and biased the picker against DT.
    for depth in range(2, enum_max_depth + 1):
        s = _strategy_for(depth, "dec_tail", 1)
        if s not in available_strategies:
            continue
        for t_large in T_LARGE_CHOICES:
            cs = cost_dec_tail_batch(workloads, c, depth=depth, t_large=t_large)
            tag = f"shared_d{depth}_dec_tail_t{t_large}"
            debug[tag] = cs
            candidates.append((s, cs, t_large, depth, 1))

    # Cross-prompt 3L dec_tail candidate — only when the batch shares a
    # cross-prompt sys prefix (cross_prompt_lca>0). The picker weighs reading
    # the shared sys ONCE (L0 = one group of all B*K beams) against the
    # per-prompt strategies that re-list sys per group; the crossover falls out
    # of cost_xprompt_dec_tail_batch's honest tile+bandwidth counting. depth=3,
    # pool_count=1 are carried for dispatch metadata (the driver rebuilds the
    # cross-prompt layout structurally).
    if cross_prompt_lca > 0 and Strategy.XPROMPT_DEC_TAIL in available_strategies:
        for t_large in T_LARGE_CHOICES:
            cs = cost_xprompt_dec_tail_batch(
                workloads, c, cross_prompt_lca=cross_prompt_lca, t_large=t_large,
            )
            debug[f"xprompt_dec_tail_t{t_large}"] = cs
            candidates.append((Strategy.XPROMPT_DEC_TAIL, cs, t_large, 3, 1))

    if available_strategies is not None:
        filtered = [cand for cand in candidates if cand[0] in available_strategies]
        if not filtered:
            # Fallback: SHARED_NL_* forced but workload doesn't support
            # depth=N (typical at the first ~16 decode steps before
            # forks settle, or when depth>3 is requested but the
            # workload has only one intermediate level). Collapse depth-N
            # picks to the deepest supported depth at the same family.
            #
            # Because the upfront filter above skipped cost evaluation
            # for non-allowed strategies, we have to re-evaluate costs
            # for the expanded set here.
            _name_to_kind = {  # strategy → (depth, family, pool_count)
                Strategy.PER_BEAM: (1, "per_beam", 1),
            }
            for d in range(2, 7):
                for pc in (1, 2):
                    name = f"SHARED_{d}L_{pc}POOL"
                    if hasattr(Strategy, name):
                        _name_to_kind[Strategy[name]] = (d, "shared", pc)
                name = f"SHARED_{d}L_DEC_TAIL"
                if hasattr(Strategy, name):
                    _name_to_kind[Strategy[name]] = (d, "dec_tail", 1)

            expanded = set(available_strategies)
            for s in list(available_strategies):
                kind = _name_to_kind.get(s)
                if kind is None or kind[1] == "per_beam":
                    continue
                d, family, pc = kind
                # Try shallower depths down to 2.
                for d_try in range(d - 1, 1, -1):
                    s_fallback = _strategy_for(d_try, family, pc)
                    expanded.add(s_fallback)
            # Re-evaluate costs for any strategy in ``expanded`` that
            # wasn't already in ``available_strategies`` (the upfront
            # filter skipped them).
            new_keys = expanded - set(available_strategies)
            for depth in range(2, enum_max_depth + 1):
                for pool_count in (1, 2):
                    s = _strategy_for(depth, "shared", pool_count)
                    if s in new_keys:
                        for t_large in T_LARGE_CHOICES:
                            cs = cost_shared_batch(
                                workloads, c,
                                depth=depth, pool_count=pool_count,
                                t_large=t_large, fused_merge=fused_merge,
                            )
                            candidates.append((s, cs, t_large, depth, pool_count))
            for depth in range(2, enum_max_depth + 1):
                s = _strategy_for(depth, "dec_tail", 1)
                if s in new_keys:
                    cs = cost_dec_tail_batch(workloads, c, depth=depth)
                    candidates.append((s, cs, 64, depth, 1))
            filtered = [cand for cand in candidates if cand[0] in expanded]
            if not filtered:
                raise ValueError(
                    "no strategy available after filtering (incl. depth-N → "
                    "shallower-depth fallbacks)"
                )
        candidates = filtered

    # DEC_TAIL switch-margin (hysteresis): inflate DEC_TAIL-family candidates'
    # *comparison* cost so DEC_TAIL wins only when cheaper than the best
    # alternative by >= dec_tail_switch_margin. The raw cost is still reported
    # in ``estimated_us`` / ``debug`` (margin affects only which candidate wins).
    margin = c.dec_tail_switch_margin
    if margin:
        best = min(
            candidates,
            key=lambda x: x[1] * (1.0 + margin) if "dec_tail" in x[0].value else x[1],
        )
    else:
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
    cross_prompt_lca: int = 0,
) -> Pick:
    """Pick the best strategy for a single prompt — wraps
    ``pick_strategy_batch`` with a 1-element list. Existing single-prompt
    callers (driver, oracle_vs_model, autotune) keep working unchanged.
    (``cross_prompt_lca`` is a no-op here: a 1-element batch has B<2 so the
    cross-prompt candidate is never generated.)
    """
    return pick_strategy_batch(
        [w], c, fused_merge=fused_merge,
        available_strategies=available_strategies,
        cross_prompt_lca=cross_prompt_lca,
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

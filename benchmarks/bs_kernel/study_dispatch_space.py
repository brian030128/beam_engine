"""Phase 0: empirical dispatch-space study.

Question: when given a much larger candidate space than the production
picker's 11, what does the cost model actually pick across realistic
tree-structured workloads?

Specifically tests:
  - depth ∈ {2, 3, 4, 5, 6}   (production: {2, 3})
  - T_large ∈ {16, 32, 64, 96, 128}  (production: {64, 128})
  - pool_count ∈ {1, 2}        (production: {1, 2}; 3-pool deferred)
  - tail_kernel ∈ {prefill, dec_tail}  (unchanged)

Workloads cover beam-search prefix-shared sibling forks at depths 2-7,
DBS group structures, and EAGLE-style speculative trees of depth 5-6.

Output: CSV with one row per cell × candidate (chosen-by-picker, plus
all alternatives for margin analysis).

This module is self-contained so it can run without changing production
``cost_model.py``. It reuses ``_level_tiles_and_bytes`` and
``Coefficients`` for the cost arithmetic.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from beam_engine.methods.bs_kernel.cost_model import (
    Coefficients,
    T_SMALL,
    _bytes_to_us,
    _ceil_div,
    _level_effective_bw_us,
    _level_tiles_and_bytes,
)


# ---------------------------------------------------------------------------
# Generalized workload representation: hierarchical sharing of arbitrary depth
# ---------------------------------------------------------------------------


@dataclass
class DeepWorkload:
    """A uniform-branching hierarchical workload.

    ``group_counts``:  [G_1, G_2, ..., G_{N-1}] with 1 < G_1 < ... < G_{N-1} <= K.
                       G_0 = 1 (root: all K beams) and G_N = K (per-beam) are
                       implicit. depth = N.
    ``shared_lens``:   [L_0, L_1, ..., L_{N-1}, L_tail] with N+1 entries.
                       L_0 is the root prefix length; L_i (0<i<N) is tokens
                       shared by each group at level i (in tokens, not pages);
                       L_tail is per-beam unique tail length.
    """
    K: int
    group_counts: list[int]
    shared_lens: list[int]
    num_kv_heads: int = 8
    head_dim: int = 128
    bytes_per_kv: int = 4096   # 2 * num_kv_heads * head_dim * dtype_bytes

    def __post_init__(self):
        assert len(self.shared_lens) == len(self.group_counts) + 2, \
            f"shared_lens has {len(self.shared_lens)} entries, expected {len(self.group_counts) + 2}"
        if self.group_counts:
            assert self.group_counts[0] >= 1
            for i in range(1, len(self.group_counts)):
                assert self.group_counts[i] > self.group_counts[i-1], \
                    f"group counts must be strictly increasing: {self.group_counts}"
            assert self.group_counts[-1] <= self.K

    @property
    def depth_max(self) -> int:
        """Maximum cascade depth possible for this workload (root + all
        intermediates + per-beam tail)."""
        return len(self.group_counts) + 2

    def levels_at_depth(self, depth: int) -> list[tuple[int, int, int, int]]:
        """Build (g_count, beams_per_group, kv_tokens, bytes_per_kv) tuples
        for a cascade of ``depth`` levels.

        depth=2: just (root + per-beam tail). All intermediates fold into tail.
        depth=N (N <= depth_max): root + N-2 intermediates + per-beam tail.
        depth > depth_max: not allowed.
        """
        assert 2 <= depth <= self.depth_max, \
            f"depth {depth} out of range [2, {self.depth_max}]"

        levels: list[tuple[int, int, int, int]] = []
        # Level 0: root, 1 group of K beams.
        levels.append((1, self.K, self.shared_lens[0], self.bytes_per_kv))

        if depth == 2:
            # Fold all intermediates + tail into per-beam.
            extra_per_beam = sum(
                self.shared_lens[i] for i in range(1, len(self.shared_lens) - 1)
            )
            tail = self.shared_lens[-1] + extra_per_beam
            levels.append((self.K, 1, tail, self.bytes_per_kv))
            return levels

        # Use the first (depth-2) intermediate levels as-is; fold deeper
        # intermediates into the per-beam tail.
        n_inter_used = depth - 2
        for i in range(n_inter_used):
            g_count = self.group_counts[i]
            beams_per_g = self.K // g_count
            inter_len = self.shared_lens[i + 1]  # shared_lens[0] is root
            levels.append((g_count, beams_per_g, inter_len, self.bytes_per_kv))

        # Per-beam tail: collapse remaining intermediates beyond depth-2 into tail.
        unused_inter_lens = sum(
            self.shared_lens[i + 1]
            for i in range(n_inter_used, len(self.group_counts))
        )
        tail = self.shared_lens[-1] + unused_inter_lens
        levels.append((self.K, 1, tail, self.bytes_per_kv))
        return levels


# ---------------------------------------------------------------------------
# Generalized cost expressions
# ---------------------------------------------------------------------------


def cost_per_beam(w: DeepWorkload, c: Coefficients, *, B: int = 1) -> float:
    """Plain paged decode: each beam reads its full KV path. For a uniform
    batch of B prompts, tile counts and bytes scale linearly with B."""
    total_kv_tokens_per_beam = sum(w.shared_lens)
    total_bytes = B * w.K * total_kv_tokens_per_beam * w.bytes_per_kv
    return _bytes_to_us(total_bytes, c)


def cost_shared(
    w: DeepWorkload,
    c: Coefficients,
    *,
    depth: int,
    pool_count: int,
    t_large: int,
    B: int = 1,
) -> float:
    """Cost of a depth-N shared-prefix cascade for a uniform batch of B prompts.

    Mirrors cost_model.cost_shared_batch: tile counts and bytes sum across
    prompts (B× single-prompt values for uniform batch), then wave count is
    computed once across the entire batch (this is the cross-batch
    SM-occupancy effect the production picker uses).
    """
    levels = w.levels_at_depth(depth)
    large_per, small_per, _bytes_per = _level_tiles_and_bytes(
        levels, pool_count=pool_count, t_large=t_large,
    )
    total_large = B * large_per
    total_small = B * small_per
    # Per-level effective BW (penalizes low-utilization prefill tiles).
    # B prompts with identical levels: each level's bytes scale by B,
    # which is equivalent to scaling the per-prompt bw_us by B.
    bw_us = B * _level_effective_bw_us(
        levels, c, pool_count=pool_count, t_large=t_large,
    )
    waves_large = max(1, _ceil_div(total_large, c.num_sms)) if total_large > 0 else 0
    waves_small = max(1, _ceil_div(total_small, c.num_sms)) if total_small > 0 else 0
    compute_us = (
        waves_large * c.per_tile_us[t_large]
        + waves_small * c.per_tile_us[T_SMALL]
    )
    extra_us = c.share_extra_us
    if pool_count == 2:
        extra_us += c.dual_pool_extra_us
    return bw_us + compute_us + extra_us


def cost_dec_tail(
    w: DeepWorkload,
    c: Coefficients,
    *,
    depth: int,
    B: int = 1,
) -> float:
    """Hybrid: prefix(/intermediate) prefill (single pool, T=64) + per-beam
    paged-decode tail + n_merges merges, for a uniform batch of B prompts.

    The prefix-side prefill cost uses per-level effective BW (penalizes
    low-utilization tiles). The tail uses the decode kernel which keeps
    peak BW (purpose-built for CTA_Q=1).
    """
    levels = w.levels_at_depth(depth)
    prefix_levels = levels[:-1]
    tail_level = levels[-1]
    large_per, _small, _prefix_bytes_per = _level_tiles_and_bytes(
        prefix_levels, pool_count=1, t_large=64,
    )
    total_large = B * large_per
    bw_us = B * _level_effective_bw_us(
        prefix_levels, c, pool_count=1, t_large=64,
    )
    waves_large = max(1, _ceil_div(total_large, c.num_sms)) if total_large > 0 else 0
    prefix_us = bw_us + waves_large * c.per_tile_us[64] + c.share_extra_us

    g_count, beams_per_g, kv_tokens, bytes_per_kv = tail_level
    n_beams = B * g_count * beams_per_g
    if kv_tokens == 0 or n_beams == 0:
        tail_us = 0.0
    else:
        tail_bytes = n_beams * kv_tokens * bytes_per_kv
        tail_bw_us = _bytes_to_us(tail_bytes, c)
        tail_work_us = n_beams * kv_tokens * c.decode_us_per_beam_kv_token
        tail_us = c.decode_launch_us + max(tail_bw_us, tail_work_us)

    n_merges = depth - 1
    return prefix_us + tail_us + n_merges * c.merge_us


# ---------------------------------------------------------------------------
# Generalized picker
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One concrete dispatch choice."""
    family: str           # "per_beam", "shared", "dec_tail"
    depth: int            # 1 for per_beam, 2..N for shared/dec_tail
    pool_count: int       # 1 for per_beam/dec_tail, 1 or 2 for shared
    t_large: int          # 16/32/64/96/128 for shared, 64 for dec_tail, 0 for per_beam

    def tag(self) -> str:
        if self.family == "per_beam":
            return "per_beam"
        if self.family == "shared":
            return f"shared_d{self.depth}_p{self.pool_count}_t{self.t_large}"
        return f"shared_d{self.depth}_dec_tail"


@dataclass
class StudyPick:
    chosen: Candidate
    chosen_us: float
    all_costs: dict[str, float]   # tag → predicted_us, for every candidate evaluated


def enumerate_candidates(
    *,
    depths: list[int],
    pool_counts: list[int],
    t_large_values: list[int],
    enable_dec_tail: bool,
) -> list[Candidate]:
    """Build the full candidate list. depths must include the maximum
    depth supported by the workloads; the picker filters per-workload."""
    out: list[Candidate] = [Candidate(family="per_beam", depth=1, pool_count=1, t_large=0)]
    for depth in depths:
        for pc in pool_counts:
            for t in t_large_values:
                out.append(Candidate(family="shared", depth=depth, pool_count=pc, t_large=t))
        if enable_dec_tail:
            out.append(Candidate(family="dec_tail", depth=depth, pool_count=1, t_large=64))
    return out


def pick_strategy(
    w: DeepWorkload,
    c: Coefficients,
    candidates: list[Candidate],
    *,
    B: int = 1,
) -> StudyPick:
    all_costs: dict[str, float] = {}
    valid: list[tuple[Candidate, float]] = []
    for cand in candidates:
        if cand.family == "per_beam":
            us = cost_per_beam(w, c, B=B)
        elif cand.family == "shared":
            if cand.depth > w.depth_max:
                continue
            us = cost_shared(
                w, c,
                depth=cand.depth, pool_count=cand.pool_count,
                t_large=cand.t_large, B=B,
            )
        else:  # dec_tail
            if cand.depth > w.depth_max:
                continue
            us = cost_dec_tail(w, c, depth=cand.depth, B=B)
        all_costs[cand.tag()] = us
        valid.append((cand, us))
    chosen_cand, chosen_us = min(valid, key=lambda x: x[1])
    return StudyPick(chosen=chosen_cand, chosen_us=chosen_us, all_costs=all_costs)


# ---------------------------------------------------------------------------
# Synthetic workload set
# ---------------------------------------------------------------------------


def _build_uniform_hierarchy(
    K: int, n_inter: int, L_p: int, inter_len: int, tail_len: int,
    *, branching: int = 2, num_kv_heads: int = 8, head_dim: int = 128,
) -> Optional[DeepWorkload]:
    """Build a uniform K-beam hierarchy with n_inter intermediate levels.

    Group counts at intermediate levels follow geometric branching:
    G_i = branching^i (clipped to K). Returns None if the resulting
    cardinalities aren't strictly increasing or exceed K.
    """
    if n_inter == 0:
        return DeepWorkload(
            K=K, group_counts=[],
            shared_lens=[L_p, tail_len],
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            bytes_per_kv=2 * num_kv_heads * head_dim * 2,
        )
    group_counts: list[int] = []
    g = branching
    for _ in range(n_inter):
        if g >= K or (group_counts and g <= group_counts[-1]):
            return None
        group_counts.append(g)
        g *= branching
    if group_counts[-1] >= K:
        return None
    shared_lens = [L_p] + [inter_len] * n_inter + [tail_len]
    return DeepWorkload(
        K=K, group_counts=group_counts, shared_lens=shared_lens,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        bytes_per_kv=2 * num_kv_heads * head_dim * 2,
    )


def build_workload_grid(
    *,
    K_values: list[int],
    L_p_values: list[int],
    n_inter_values: list[int],
    inter_len_values: list[int],
    tail_len_values: list[int],
    branching_values: list[int],
) -> list[tuple[str, DeepWorkload]]:
    """Cartesian-product workload grid. Each entry is (label, workload)."""
    out: list[tuple[str, DeepWorkload]] = []
    for K in K_values:
        for L_p in L_p_values:
            for n_inter in n_inter_values:
                for inter_len in inter_len_values:
                    for tail_len in tail_len_values:
                        for branching in branching_values:
                            w = _build_uniform_hierarchy(
                                K=K, n_inter=n_inter, L_p=L_p,
                                inter_len=inter_len, tail_len=tail_len,
                                branching=branching,
                            )
                            if w is None:
                                continue
                            depth_max = w.depth_max
                            label = (
                                f"K{K}_Lp{L_p}_nI{n_inter}_Li{inter_len}"
                                f"_Lt{tail_len}_br{branching}_dm{depth_max}"
                            )
                            out.append((label, w))
    return out


# ---------------------------------------------------------------------------
# Sweep + report
# ---------------------------------------------------------------------------


def run_study(
    workloads: list[tuple[str, DeepWorkload]],
    candidates: list[Candidate],
    coeffs: Coefficients,
    out_csv: Path,
    *,
    B_values: list[int] = (1,),
) -> dict:
    """Run the picker over every (workload, B, candidate) cell. Write the
    full predicted-µs vector to CSV. Return summary stats.
    """
    rows: list[dict] = []
    chosen_strategy_counts: dict[str, int] = {}
    chosen_pool_counts: dict[int, int] = {}
    max_depth_chosen = 0
    cells_where_new_T_chosen = 0
    cells_where_deep_chosen = 0  # depth > 3
    cells_where_2pool_chosen = 0
    cells_where_dec_tail_chosen = 0
    PRODUCTION_T = {64, 128}

    for label, w in workloads:
        for B in B_values:
            pick = pick_strategy(w, coeffs, candidates, B=B)
            chosen_tag = pick.chosen.tag()
            chosen_strategy_counts[chosen_tag] = chosen_strategy_counts.get(chosen_tag, 0) + 1
            chosen_pool_counts[pick.chosen.pool_count] = chosen_pool_counts.get(pick.chosen.pool_count, 0) + 1
            max_depth_chosen = max(max_depth_chosen, pick.chosen.depth)
            if pick.chosen.family == "shared" and pick.chosen.t_large not in PRODUCTION_T:
                cells_where_new_T_chosen += 1
            if pick.chosen.depth > 3:
                cells_where_deep_chosen += 1
            if pick.chosen.pool_count == 2:
                cells_where_2pool_chosen += 1
            if pick.chosen.family == "dec_tail":
                cells_where_dec_tail_chosen += 1

            prod_costs = {
                tag: us for tag, us in pick.all_costs.items()
                if _is_in_production_11(tag)
            }
            if prod_costs:
                best_prod_tag = min(prod_costs, key=prod_costs.get)
                best_prod_us = prod_costs[best_prod_tag]
                margin_vs_prod = (best_prod_us / pick.chosen_us) - 1.0
            else:
                best_prod_tag = ""
                best_prod_us = math.inf
                margin_vs_prod = math.inf

            row = {
                "label": label,
                "B": B,
                "K": w.K,
                "L_p": w.shared_lens[0],
                "depth_max": w.depth_max,
                "group_counts": "/".join(str(g) for g in w.group_counts),
                "chosen_tag": chosen_tag,
                "chosen_us": f"{pick.chosen_us:.3f}",
                "chosen_depth": pick.chosen.depth,
                "chosen_t_large": pick.chosen.t_large,
                "chosen_pool": pick.chosen.pool_count,
                "chosen_family": pick.chosen.family,
                "best_in_prod_11_tag": best_prod_tag,
                "best_in_prod_11_us": f"{best_prod_us:.3f}" if best_prod_us != math.inf else "",
                "prod_11_relative_loss": f"{margin_vs_prod:.4f}" if margin_vs_prod != math.inf else "",
                "all_costs_json": json.dumps({k: round(v, 3) for k, v in pick.all_costs.items()}),
            }
            rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_cells": len(rows),
        "B_values": list(B_values),
        "max_depth_chosen": max_depth_chosen,
        "cells_where_deep_chosen": cells_where_deep_chosen,
        "cells_where_new_T_chosen": cells_where_new_T_chosen,
        "cells_where_2pool_chosen": cells_where_2pool_chosen,
        "cells_where_dec_tail_chosen": cells_where_dec_tail_chosen,
        "chosen_pool_counts": chosen_pool_counts,
        "chosen_strategy_counts": dict(sorted(
            chosen_strategy_counts.items(), key=lambda x: -x[1]
        )),
    }
    return summary


def _is_in_production_11(tag: str) -> bool:
    """Check whether a candidate tag corresponds to one of the production-11
    strategies. Production set:
      - per_beam
      - shared_d{2,3}_p{1,2}_t{64,128}
      - shared_d{2,3}_dec_tail
    """
    if tag == "per_beam":
        return True
    if tag in {"shared_d2_dec_tail", "shared_d3_dec_tail"}:
        return True
    parts = tag.split("_")
    # shared_d<N>_p<M>_t<T>
    if len(parts) == 4 and parts[0] == "shared" and parts[2].startswith("p") and parts[3].startswith("t"):
        try:
            depth = int(parts[1][1:])
            pool = int(parts[2][1:])
            t = int(parts[3][1:])
        except ValueError:
            return False
        return depth in {2, 3} and pool in {1, 2} and t in {64, 128}
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


DEFAULT_K_VALUES = [16, 32, 64]
DEFAULT_L_P_VALUES = [1024, 4096, 8192, 32768]
DEFAULT_N_INTER_VALUES = [0, 1, 2, 3, 4]   # depth_max ∈ {2, 3, 4, 5, 6}
DEFAULT_INTER_LEN_VALUES = [64, 256, 512]
DEFAULT_TAIL_LEN_VALUES = [16, 64, 256]
DEFAULT_BRANCHING_VALUES = [2, 4]


def _load_calibrated_coeffs() -> Coefficients:
    """Load H100 calibrated coefficients if cached; fall back to defaults.

    Adds linearly-interpolated per_tile_us values for T=32 and T=96 since
    the production cost model only calibrates {16, 64, 128}. The
    interpolation is documented in the findings report; if A4
    surfaces T∈{32, 96} as picker choices, those need real calibration.
    """
    cache_dir = Path.home() / ".cache" / "beam_engine"
    coeffs_files = sorted(cache_dir.glob("coeffs-*.json"))
    coeffs_files = [p for p in coeffs_files if not p.name.endswith(".bak")]
    if not coeffs_files:
        print("[study] No cached coefficients; using Coefficients.defaults()")
        return Coefficients.defaults()
    coeffs_file = coeffs_files[0]
    print(f"[study] Loading calibrated coefficients: {coeffs_file.name}")
    data = json.loads(coeffs_file.read_text())

    per_tile = {int(k): float(v) for k, v in data.get("per_tile_us", {}).items()}
    # Linear interpolation for T=32, T=96 (not calibrated in production).
    if 32 not in per_tile and 16 in per_tile and 64 in per_tile:
        per_tile[32] = per_tile[16] + (per_tile[64] - per_tile[16]) * (32 - 16) / (64 - 16)
    if 96 not in per_tile and 64 in per_tile and 128 in per_tile:
        per_tile[96] = per_tile[64] + (per_tile[128] - per_tile[64]) * (96 - 64) / (128 - 64)

    c = Coefficients(
        B_hbm=data.get("B_hbm", 1_000_000.0),
        launch_us=data.get("launch_us", 6.0),
        sync_us=data.get("sync_us", 4.0),
        merge_us=data.get("merge_us", 2.0),
        per_tile_us=per_tile,
        per_beam_us_per_kv_token=data.get("per_beam_us_per_kv_token", 0.0008),
        num_sms=data.get("num_sms", 84),
        share_extra_us=data.get("share_extra_us", 0.0),
        dual_pool_extra_us=data.get("dual_pool_extra_us", 0.0),
        decode_us_per_beam_kv_token=data.get("decode_us_per_beam_kv_token", 0.0008),
        decode_launch_us=data.get("decode_launch_us", 2.0),
        # H100 fit from `bench_dispatch_grid` measurements (16 cells across
        # K∈{16,64} × L_p∈{2K,8K,32K} × B∈{1,8,32}, Llama-3.2-1B):
        # floor=0.5 reproduces the measured per-cell winning strategy
        # on 13/16 cells (81% match rate) with mean regret 0.4% and
        # max regret 2.8%. The penalty applies only at low-util,
        # high-tile-count levels (per-beam-tail-like situations,
        # captured by `g_count >= 8 AND util < 0.5` in
        # `_level_effective_bw_us`); root level with packed=K and
        # tile_count=1 gets no penalty.
        bw_efficiency_floor=data.get("bw_efficiency_floor", 0.5),
    )
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("benchmarks/bs_kernel/results/dispatch_space_study"))
    parser.add_argument("--depth-max", type=int, default=6,
                        help="max depth to enumerate as candidate")
    parser.add_argument("--include-new-T", action="store_true", default=True,
                        help="include T_large ∈ {16, 32, 96} alongside {64, 128}")
    parser.add_argument("--use-calibrated", action="store_true", default=True,
                        help="load calibrated coefficients from ~/.cache/beam_engine/")
    parser.add_argument("--B-values", type=int, nargs="+", default=[1, 8, 32],
                        help="batch sizes to sweep (uniform batches of B identical prompts)")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir.parent / f"{args.out_dir.name}-{timestamp}"
    out_csv = out_dir / "study.csv"

    # Build candidate space.
    depths = list(range(2, args.depth_max + 1))
    pool_counts = [1, 2]
    t_values = [16, 32, 64, 96, 128] if args.include_new_T else [64, 128]
    candidates = enumerate_candidates(
        depths=depths,
        pool_counts=pool_counts,
        t_large_values=t_values,
        enable_dec_tail=True,
    )
    print(f"[study] Candidate count: {len(candidates)}")

    # Build workload grid.
    workloads = build_workload_grid(
        K_values=DEFAULT_K_VALUES,
        L_p_values=DEFAULT_L_P_VALUES,
        n_inter_values=DEFAULT_N_INTER_VALUES,
        inter_len_values=DEFAULT_INTER_LEN_VALUES,
        tail_len_values=DEFAULT_TAIL_LEN_VALUES,
        branching_values=DEFAULT_BRANCHING_VALUES,
    )
    print(f"[study] Workload grid: {len(workloads)} cells")

    coeffs = _load_calibrated_coeffs() if args.use_calibrated else Coefficients.defaults()
    print(f"[study] share_extra_us={coeffs.share_extra_us}, "
          f"dual_pool_extra_us={coeffs.dual_pool_extra_us}, "
          f"merge_us={coeffs.merge_us}, num_sms={coeffs.num_sms}")
    print(f"[study] per_tile_us={coeffs.per_tile_us}")
    print(f"[study] B values: {args.B_values}")
    summary = run_study(workloads, candidates, coeffs, out_csv, B_values=args.B_values)
    summary["candidate_count"] = len(candidates)
    summary["t_values"] = t_values
    summary["depths"] = depths
    summary["pool_counts"] = pool_counts

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[study] CSV: {out_csv}")
    print(f"[study] Summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

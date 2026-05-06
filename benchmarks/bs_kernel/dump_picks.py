"""Run bs_kernel at chosen (K, L_p) cells and dump:
  1. The strategy histogram (which Pick was made each step).
  2. The CTA_Q tile-pool layout per pick.
  3. Per-tile padding % per level (waste = tile_size - packed_queries).

Usage:
    uv run python benchmarks/bs_kernel/dump_picks.py \\
        --K 16 32 64 --L_p 51200 --max_new 16
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from math import ceil

import torch
from transformers import AutoTokenizer

from beam_engine.methods.bs_kernel import beam_search as bs_kernel_search
from beam_engine.methods.bs_kernel.cost_model import T_LARGE_CHOICES, T_SMALL, Strategy
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16


def _make_prompt(tok, target_len: int) -> list[int]:
    """Repeat a seed prompt until at least target_len tokens, then truncate."""
    seed = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar named Elara who studied the stars and the ways of the "
        "natural world. She traveled across mountains and rivers, "
        "recording her observations in a leather-bound journal. "
    )
    ids: list[int] = []
    while len(ids) < target_len:
        ids.extend(tok.encode(seed, add_special_tokens=False))
    return ids[:target_len]


def padding_for_level(g_count: int, packed: int, t_used: int) -> tuple[int, int, float]:
    """Return (tiles, real_queries, padding_pct) for a level."""
    tiles = max(1, ceil(packed / t_used))
    cells = tiles * t_used  # per group
    waste = cells - packed
    # aggregate across all groups in the level
    total_cells = g_count * cells
    total_real = g_count * packed
    pct = 100.0 * (total_cells - total_real) / total_cells if total_cells else 0.0
    return g_count * tiles, total_real, pct


def analyze_pick(pick, K: int, L_p: int, suffix_lens: list[int], inter=None):
    """Compute per-level (T, real_q, padding_pct) for one Pick."""
    if pick.strategy == Strategy.PER_BEAM:
        # paged decode: K independent sequences, no Q-tiling padding to speak of
        return [("per_beam_paged", T_SMALL, K, K, 0.0)]

    t_large = pick.t_large
    pool = pick.pool_count
    depth = pick.depth
    avg_suffix = sum(suffix_lens) / len(suffix_lens) if suffix_lens else 0
    rows = []

    # Level 0: shared, g=1, B=K
    if pool == 1:
        T0 = t_large
    else:
        T0 = t_large if K > T_SMALL else T_SMALL
    tiles0, real0, pct0 = padding_for_level(1, K, T0)
    rows.append(("L0_shared", T0, K, tiles0, pct0))

    # Level mid (depth=3): g=G, B=K/G, kv=inter_len
    if depth == 3 and inter is not None:
        G = inter.G
        B_mid = inter.group_size
        if pool == 1:
            Tm = t_large
        else:
            Tm = t_large if B_mid > T_SMALL else T_SMALL
        tilesm, realm, pctm = padding_for_level(G, B_mid, Tm)
        rows.append(("L1_inter", Tm, G * B_mid, tilesm, pctm))

    # Last level: K groups of 1 beam each
    if pool == 1:
        Tl = t_large
    else:
        Tl = T_SMALL  # B=1 ≤ T_SMALL
    tilesl, reall, pctl = padding_for_level(K, 1, Tl)
    rows.append(("L_per_beam", Tl, K, tilesl, pctl))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--L_p", nargs="+", type=int, default=[51200])
    ap.add_argument("--max_new", type=int, default=16)
    args = ap.parse_args()

    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    for K in args.K:
        for L_p in args.L_p:
            print(f"=" * 80)
            print(f"K={K}, L_p={L_p}, max_new={args.max_new}")
            print(f"=" * 80)
            prompt = _make_prompt(tok, L_p)
            beams, timings, picks = bs_kernel_search(
                model, config, [prompt], args.max_new, K,
                return_timings=True, return_picks=True,
                max_num_pages=8192,  # L_p=51200 / page_size=16 = 3200 pages just for prompt
            )
            picks_per_prompt = picks[0]  # 1 prompt
            # Strategy histogram
            strat_hist = Counter(p.strategy.value for p in picks_per_prompt)
            print(f"\nStrategy histogram across {len(picks_per_prompt)} decode steps:")
            for s, n in strat_hist.most_common():
                print(f"  {s:<22} {n:>3}")
            # t_large + pool histogram (for SHARED only)
            shared_hist = Counter(
                (p.t_large, p.pool_count, p.depth)
                for p in picks_per_prompt if p.strategy != Strategy.PER_BEAM
            )
            if shared_hist:
                print(f"\n(t_large, pool_count, depth) histogram for SHARED steps:")
                for k, n in shared_hist.most_common():
                    print(f"  t_large={k[0]:<3}  pool={k[1]}  depth={k[2]}   {n:>3}")
            # CTA_Q padding % per level, averaged across steps
            level_padding: dict[str, list[float]] = defaultdict(list)
            level_tiles: dict[str, list[int]] = defaultdict(list)
            level_T: dict[str, list[int]] = defaultdict(list)
            # Reconstruct workload-ish suffix_lens (uniform avg = step_idx + 1)
            for step_idx, p in enumerate(picks_per_prompt):
                # suffix at step s: each beam has appended s+1 tokens of decode
                avg_tail = step_idx + 1
                rows = analyze_pick(p, K, L_p, [avg_tail] * K, inter=None)
                for name, T, real, tiles, pct in rows:
                    level_padding[name].append(pct)
                    level_tiles[name].append(tiles)
                    level_T[name].append(T)
            print(f"\nPer-level CTA_Q padding (mean across decode steps):")
            print(f"  {'level':<14} {'T_used':>8} {'tiles':>8} {'padding%':>10}")
            for name in level_padding:
                pads = level_padding[name]
                tiles = level_tiles[name]
                Ts = level_T[name]
                T_mode = Counter(Ts).most_common(1)[0][0]
                print(
                    f"  {name:<14} {T_mode:>8} {sum(tiles)/len(tiles):>8.1f} "
                    f"{sum(pads)/len(pads):>9.1f}%"
                )
            print()


if __name__ == "__main__":
    main()

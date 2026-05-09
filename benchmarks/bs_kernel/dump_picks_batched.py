"""Run bs_kernel at chosen (K, L_p, B) cells and dump:
  1. The strategy histogram across decode steps.
  2. The (t_large, pool_count, depth) histogram for SHARED picks.
  3. Per-level CTA_Q padding %, accounting for the B-prompt batched layout
     (B groups at level 0, B*K singletons at level 1).

Usage:
    uv run python benchmarks/bs_kernel/dump_picks_batched.py \\
        --K 16 32 64 --L_p 8192 --B 1 2 4 8 --max_new 16
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from math import ceil

import torch
from transformers import AutoTokenizer

from beam_engine.methods.bs_kernel import beam_search as bs_kernel_search
from beam_engine.methods.bs_kernel.cost_model import T_LARGE_CHOICES, T_SMALL, Strategy
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda"
DTYPE = torch.float16


def _make_prompt(tok, target_len: int) -> list[int]:
    seed = (
        "Once upon a time, in a kingdom far away, there lived a curious "
        "scholar named Elara who studied the stars and the ways of the "
        "natural world. "
    )
    ids: list[int] = []
    while len(ids) < target_len:
        ids.extend(tok.encode(seed, add_special_tokens=False))
    return ids[:target_len]


_DISTINCT_SEEDS = [
    "Once upon a time, in a kingdom far away, there lived a curious scholar who studied the stars. ",
    "The bustling marketplace of Constantinople was filled with merchants from every corner of the known world. ",
    "Deep in the Amazon rainforest, biologists discovered a new species of luminescent frog. ",
    "On the dusty plains of the American Midwest, a young farmer dreamed of becoming a railroad engineer. ",
    "In Edo-period Japan, a master swordsmith forged blades that were said to sing when drawn. ",
    "The Antarctic research station crackled with radio static as the long polar night began. ",
    "Aboard the steamship bound for Liverpool, the elderly diplomat reread the encrypted message. ",
    "High in the Andes, a llama herder noticed strange patterns carved into the volcanic rock. ",
    "The detective lit her pipe and stared at the rain streaking down the office window. ",
    "Long before the first cities rose along the Tigris, hunter-gatherers followed seasonal herds. ",
    "When the great library of Alexandria still stood, scholars argued about the shape of the heavens. ",
    "The Viking longship cut through the cold North Sea waters under a sky heavy with gulls. ",
    "In a sleepy New England fishing village, the lighthouse keeper recorded the day's tide. ",
    "Beneath the towering canopy of redwoods, a paleobotanist sifted through compressed peat. ",
    "The neon-lit streets of Shibuya pulsed with commuters hurrying through the spring drizzle. ",
    "Aboard the International Space Station, the flight engineer floated past the Cupola windows. ",
    "In the dim lamplight of the medieval scriptorium, the monk dipped his quill once more. ",
    "The desert caravan paused at the oasis as the sun began its descent toward the dunes. ",
    "On a quiet research vessel in the Pacific, the marine biologist tagged her hundredth manta ray. ",
    "Through the crowded bazaar of old Damascus, the spice merchant called out his prices. ",
    "The astronaut adjusted her helmet visor and stepped onto the regolith of the lunar far side. ",
    "Inside the cathedral, the organist practiced a fugue while the masons repaired the buttress. ",
    "The Mongolian steppes stretched endlessly under a sky so wide it felt like another ocean. ",
    "Far below the surface of Europa, autonomous probes mapped the hydrothermal vents. ",
    "The Parisian café hummed with conversation about the latest exhibition at the Salon. ",
    "On a rocky coast of Cornwall, the lighthouse keeper recounted tales of shipwrecks past. ",
    "The Silk Road merchant unloaded his bolts of silk at the gates of Samarkand. ",
    "In the Brazilian favela, a young girl practiced her violin on the rooftop each evening. ",
    "The cartographer unrolled his maps and pointed to a coastline no European had yet seen. ",
    "Deep in the Carpathian mountains, the wolf packs moved silently through the winter snow. ",
    "Beneath the Antarctic ice shelf, the autonomous submarine recorded a never-before-heard call. ",
    "The royal astronomer of the Mughal court adjusted his sextant and watched Jupiter rise. ",
]


def _make_distinct_prompts(tok, target_len: int, B: int) -> list[list[int]]:
    prompts: list[list[int]] = []
    for i in range(B):
        seed = f"Document {i:03d}. " + _DISTINCT_SEEDS[i % len(_DISTINCT_SEEDS)]
        ids: list[int] = []
        while len(ids) < target_len:
            ids.extend(tok.encode(seed, add_special_tokens=False))
        prompts.append(ids[:target_len])
    return prompts


def padding_pct(num_groups: int, queries_per_group: int, t_used: int) -> tuple[int, float]:
    """Return (total_tiles, padding_pct) for a level with `num_groups` of
    `queries_per_group` queries, packed into tiles of size t_used."""
    tiles_per_group = max(1, ceil(queries_per_group / t_used))
    tiles = num_groups * tiles_per_group
    cells = tiles * t_used
    real = num_groups * queries_per_group
    pct = 100.0 * (cells - real) / cells if cells else 0.0
    return tiles, pct


def analyze_pick_batched(pick, K: int, B: int) -> list[tuple[str, int, int, int, float]]:
    """For one Pick covering B prompts, compute per-level (T, total_real_q,
    total_tiles, padding_pct).

    Layout in the batched cascade:
      * level 0: B groups, each of K queries (one per beam in that prompt).
      * level 1: B*K singleton groups, 1 query each.
    """
    if pick.strategy == Strategy.PER_BEAM:
        return [("per_beam_paged", T_SMALL, B * K, B * K, 0.0)]

    t_large = pick.t_large
    pool = pick.pool_count

    # Level 0: B groups of K queries.
    if pool == 1:
        T0 = t_large
    else:
        T0 = t_large if K > T_SMALL else T_SMALL
    tiles0, pct0 = padding_pct(B, K, T0)
    rows = [("L0_shared", T0, B * K, tiles0, pct0)]

    # Last level: B*K groups of 1 query.
    if pool == 1:
        Tl = t_large
    else:
        Tl = T_SMALL  # B=1 group ≤ T_SMALL
    tiles_last, pct_last = padding_pct(B * K, 1, Tl)
    rows.append(("L_per_beam", Tl, B * K, tiles_last, pct_last))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--L_p", nargs="+", type=int, default=[8192])
    ap.add_argument("--B", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--distinct", action="store_true",
                    help="use B pairwise-distinct prompts instead of B copies")
    args = ap.parse_args()

    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    for K in args.K:
        for L_p in args.L_p:
            for B in args.B:
                print(f"=" * 80)
                print(f"K={K}  L_p={L_p}  B={B}  max_new={args.max_new}")
                print(f"=" * 80)
                if args.distinct:
                    prompts = _make_distinct_prompts(tok, L_p, B)
                else:
                    base = _make_prompt(tok, L_p)
                    prompts = [list(base) for _ in range(B)]
                # max_num_pages must fit B prompts + B*K decode beams.
                ps = 16
                pages = (
                    sum((len(p) + ps - 1) // ps for p in prompts)
                    + B * K * ((args.max_new + ps - 1) // ps + 2)
                    + 256
                )
                _, _, picks = bs_kernel_search(
                    model, config, prompts, args.max_new, K,
                    return_timings=True, return_picks=True,
                    max_num_pages=pages,
                )
                # All B prompts share the same pick per step (batched picker).
                picks_steps = picks[0]
                strat_hist = Counter(p.strategy.value for p in picks_steps)
                print(f"\nStrategy histogram across {len(picks_steps)} steps:")
                for s, n in strat_hist.most_common():
                    print(f"  {s:<22} {n:>3}")
                shared_hist = Counter(
                    (p.t_large, p.pool_count, p.depth)
                    for p in picks_steps if p.strategy != Strategy.PER_BEAM
                )
                if shared_hist:
                    print(f"\n(t_large, pool_count, depth) histogram for SHARED steps:")
                    for k, n in shared_hist.most_common():
                        print(
                            f"  t_large={k[0]:<3}  pool={k[1]}  depth={k[2]}   {n:>3}"
                        )
                level_padding: dict[str, list[float]] = defaultdict(list)
                level_tiles: dict[str, list[int]] = defaultdict(list)
                level_T: dict[str, list[int]] = defaultdict(list)
                for p in picks_steps:
                    rows = analyze_pick_batched(p, K, B)
                    for name, T, real, tiles, pct in rows:
                        level_padding[name].append(pct)
                        level_tiles[name].append(tiles)
                        level_T[name].append(T)
                print(f"\nPer-level CTA_Q padding (mean across decode steps):")
                print(
                    f"  {'level':<14} {'T_used':>8} {'total_tiles':>12} "
                    f"{'tiles/wave':>11} {'padding%':>10}"
                )
                num_sms = 132  # H100
                for name in level_padding:
                    pads = level_padding[name]
                    tiles = level_tiles[name]
                    Ts = level_T[name]
                    T_mode = Counter(Ts).most_common(1)[0][0]
                    avg_tiles = sum(tiles) / len(tiles)
                    waves = avg_tiles / num_sms
                    print(
                        f"  {name:<14} {T_mode:>8} {avg_tiles:>12.1f} "
                        f"{waves:>11.2f} {sum(pads) / len(pads):>9.1f}%"
                    )
                print()


if __name__ == "__main__":
    main()

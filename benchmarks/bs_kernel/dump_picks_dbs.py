"""Same as dump_picks_batched but routes through DBS — confirm the
hypothesis that DBS's diverse forks suppress depth=3 picks because the
K beams no longer cluster into uniform G groups.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from math import ceil

import torch
from transformers import AutoTokenizer

from beam_engine.baselines.dbs import dbs_bs_kernel
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", nargs="+", type=int, default=[16, 64])
    ap.add_argument("--L_p", nargs="+", type=int, default=[8192])
    ap.add_argument("--B", nargs="+", type=int, default=[1, 16, 32])
    ap.add_argument("--max_new", type=int, default=512)
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
                print(f"DBS  K={K}  L_p={L_p}  B={B}  max_new={args.max_new}")
                print(f"=" * 80)
                base = _make_prompt(tok, L_p)
                prompts = [list(base) for _ in range(B)]
                ps = 16
                pages = (
                    sum((len(p) + ps - 1) // ps for p in prompts)
                    + B * K * ((args.max_new + ps - 1) // ps + 2)
                    + 256
                )
                _, _, picks = dbs_bs_kernel(
                    model, config, prompts, args.max_new, K,
                    return_timings=True, return_picks=True,
                    max_num_pages=pages,
                )
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
                    print(f"\n(t_large, pool_count, depth) histogram:")
                    for k, n in shared_hist.most_common():
                        print(
                            f"  t_large={k[0]:<3}  pool={k[1]}  depth={k[2]}   {n:>3}"
                        )
                print()


if __name__ == "__main__":
    main()

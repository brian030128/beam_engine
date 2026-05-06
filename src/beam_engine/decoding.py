"""Top-K selection strategies for beam search.

The attention methods don't bake in the top-K selection rule — they call
``select_at_prefill(log_probs, K)`` and ``select_at_decode(scores, K)``
on a per-prompt tensor and let the strategy decide.

The default is standard beam search top-K. Pass the (prefill, decode) pair
returned by ``dbs_strategy(num_groups, diversity_strength)`` to get Diverse
Beam Search on top of any attention kernel.

Contract — both functions operate on a *single* prompt's tensor (no batch
dim). The driving methods loop over prompts themselves.

  ``select_at_prefill(log_probs: (V,), K) -> (top_lp: (K,), top_ids: (K,))``
  ``select_at_decode(scores: (K_beams, V), K) -> (parents: (K,), tokens: (K,), top_scores: (K,))``
"""

from __future__ import annotations

from typing import Callable

import torch


PrefillSelect = Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]
DecodeSelect = Callable[
    [torch.Tensor, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


def standard_prefill_select(log_probs: torch.Tensor, K: int):
    return log_probs.topk(K, dim=-1)


def standard_decode_select(scores: torch.Tensor, K: int):
    _, V = scores.shape
    flat = scores.reshape(-1)
    top_scores, top_flat = flat.topk(K)
    parents = top_flat // V
    tokens = top_flat % V
    return parents, tokens, top_scores


def dbs_strategy(num_groups: int, diversity_strength: float):
    """Hamming-diversity Diverse Beam Search (Vijayakumar et al. 2016).

    Returns ``(prefill_fn, decode_fn)`` matching the strategy contract.
    ``num_groups`` must divide K. ``diversity_strength`` is the λ factor:
    each token's score is reduced by λ × (#earlier groups that picked it
    at this same time-step) before the group's own top-K/G.
    """
    def prefill(log_probs: torch.Tensor, K: int):
        if K % num_groups != 0:
            raise ValueError(
                f"num_groups={num_groups} must divide K={K}"
            )
        Kg = K // num_groups
        out_lp = torch.empty(K, device=log_probs.device, dtype=log_probs.dtype)
        out_ids = torch.empty(K, device=log_probs.device, dtype=torch.long)
        row = log_probs.clone()
        for g in range(num_groups):
            lp_g, ids_g = row.topk(Kg)
            out_lp[g * Kg : (g + 1) * Kg] = lp_g
            out_ids[g * Kg : (g + 1) * Kg] = ids_g
            if diversity_strength > 0 and g < num_groups - 1:
                row[ids_g] = row[ids_g] - diversity_strength
        return out_lp, out_ids

    def decode(scores: torch.Tensor, K: int):
        if K % num_groups != 0:
            raise ValueError(
                f"num_groups={num_groups} must divide K={K}"
            )
        Kg = K // num_groups
        _, V = scores.shape
        out_scores = torch.empty(K, device=scores.device, dtype=scores.dtype)
        out_parents = torch.empty(K, device=scores.device, dtype=torch.long)
        out_tokens = torch.empty(K, device=scores.device, dtype=torch.long)
        chosen_tokens: list[int] = []
        for g in range(num_groups):
            grp = scores[g * Kg : (g + 1) * Kg].clone()
            if diversity_strength > 0 and chosen_tokens:
                for tok in chosen_tokens:
                    grp[:, tok] -= diversity_strength
            top_scores, top_flat = grp.reshape(-1).topk(Kg)
            local_parent = top_flat // V
            local_token = top_flat % V
            global_parent = local_parent + g * Kg
            sl = slice(g * Kg, (g + 1) * Kg)
            out_scores[sl] = top_scores
            out_parents[sl] = global_parent
            out_tokens[sl] = local_token
            chosen_tokens.extend(local_token.tolist())
        return out_parents, out_tokens, out_scores

    return prefill, decode

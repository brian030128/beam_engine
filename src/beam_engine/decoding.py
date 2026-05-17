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


def standard_decode_select_batched(scores: torch.Tensor, K: int):
    """Batched form of ``standard_decode_select``.

    ``scores`` is shaped ``(B, K_beams, V)``. Returns three ``(B, K)``
    tensors. Drivers that loop per-prompt with the standard top-K should
    detect ``select_at_decode is standard_decode_select`` and call this
    instead — collapses 3*B device-to-host ``.tolist()`` syncs to 3.
    """
    B, _, V = scores.shape
    flat = scores.reshape(B, -1)
    top_scores, top_flat = flat.topk(K, dim=-1)
    parents = top_flat // V
    tokens = top_flat % V
    return parents, tokens, top_scores


def dbs_strategy(num_groups: int, diversity_strength: float):
    """Hamming-diversity Diverse Beam Search (Vijayakumar et al. 2016).

    Returns ``(prefill_fn, decode_fn)`` matching the strategy contract.
    ``num_groups`` must divide K. ``diversity_strength`` is the λ factor:
    each token's score is reduced by λ × (#earlier groups that picked it
    at this same time-step) before the group's own top-K/G.

    Implementation note: the penalty is accumulated into a single (V,)
    tensor between groups via ``scatter_add_``, then broadcast-added into
    the next group's scores. This avoids the per-chosen-token in-place
    kernel launches and the per-group ``.tolist()`` host sync from the
    naive implementation — both cost ~24 ms/step at B=16, K=32, G=4.
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
            sl = slice(g * Kg, (g + 1) * Kg)
            out_lp[sl] = lp_g
            out_ids[sl] = ids_g
            if diversity_strength > 0 and g < num_groups - 1:
                row.scatter_add_(
                    0, ids_g,
                    torch.full_like(lp_g, -diversity_strength),
                )
        return out_lp, out_ids

    def decode(scores: torch.Tensor, K: int):
        if K % num_groups != 0:
            raise ValueError(
                f"num_groups={num_groups} must divide K={K}"
            )
        Kg = K // num_groups
        _, V = scores.shape
        dev = scores.device
        dt = scores.dtype
        out_scores = torch.empty(K, device=dev, dtype=dt)
        out_parents = torch.empty(K, device=dev, dtype=torch.long)
        out_tokens = torch.empty(K, device=dev, dtype=torch.long)
        # Accumulated diversity penalty over all earlier groups' chosen
        # tokens. Lazily allocated — only paid for if λ > 0.
        penalty: torch.Tensor | None = None
        for g in range(num_groups):
            grp = scores[g * Kg : (g + 1) * Kg]
            if penalty is not None:
                grp = grp + penalty  # broadcast (Kg, V) + (V,)
            top_scores, top_flat = grp.reshape(-1).topk(Kg)
            local_parent = top_flat // V
            local_token = top_flat % V
            global_parent = local_parent + g * Kg
            sl = slice(g * Kg, (g + 1) * Kg)
            out_scores[sl] = top_scores
            out_parents[sl] = global_parent
            out_tokens[sl] = local_token
            if diversity_strength > 0 and g < num_groups - 1:
                if penalty is None:
                    penalty = torch.zeros(V, device=dev, dtype=dt)
                penalty.scatter_add_(
                    0, local_token,
                    torch.full_like(top_scores, -diversity_strength),
                )
        return out_parents, out_tokens, out_scores

    def decode_batched(scores: torch.Tensor, K: int):
        """Batched form of ``decode``.

        ``scores`` is shaped ``(B, K_beams, V)``. Returns three ``(B, K)``
        tensors. The G groups still loop in Python (each carries
        cumulative diversity-penalty state), but the *prompt* dimension
        is fully batched: G batched ``topk`` launches per step instead
        of B·G, and one batched ``scatter_add_`` instead of B per group.

        Drivers should detect ``select_at_decode is decode`` (i.e.
        ``getattr(select_at_decode, '_batched_form', None)``) and call
        the batched form when B > 1.
        """
        if K % num_groups != 0:
            raise ValueError(
                f"num_groups={num_groups} must divide K={K}"
            )
        Kg = K // num_groups
        B, _, V = scores.shape
        dev = scores.device
        dt = scores.dtype
        out_scores = torch.empty(B, K, device=dev, dtype=dt)
        out_parents = torch.empty(B, K, device=dev, dtype=torch.long)
        out_tokens = torch.empty(B, K, device=dev, dtype=torch.long)
        penalty: torch.Tensor | None = None  # (B, V), lazy
        for g in range(num_groups):
            grp = scores[:, g * Kg : (g + 1) * Kg, :]  # (B, Kg, V)
            if penalty is not None:
                grp = grp + penalty.unsqueeze(1)  # (B,Kg,V) + (B,1,V)
            top_scores, top_flat = grp.reshape(B, -1).topk(Kg, dim=-1)
            local_parent = top_flat // V          # (B, Kg)
            local_token = top_flat % V            # (B, Kg)
            global_parent = local_parent + g * Kg
            sl = slice(g * Kg, (g + 1) * Kg)
            out_scores[:, sl] = top_scores
            out_parents[:, sl] = global_parent
            out_tokens[:, sl] = local_token
            if diversity_strength > 0 and g < num_groups - 1:
                if penalty is None:
                    penalty = torch.zeros(B, V, device=dev, dtype=dt)
                penalty.scatter_add_(
                    1, local_token,
                    torch.full_like(top_scores, -diversity_strength),
                )
        return out_parents, out_tokens, out_scores

    decode._batched_form = decode_batched  # type: ignore[attr-defined]
    return prefill, decode

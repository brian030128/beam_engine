"""Baseline 2 — Tree-attention beam search (arXiv:2502.00085).

Within a prompt, all beams' KV is packed into one contiguous fused buffer
(prefix once, then each beam's suffix tokens). Each decode step issues a
single FlashAttention call: K queries (one per beam) against the full fused
KV, with a tree-shaped custom mask that lets each query see only the keys on
its root-to-leaf path. The shared prefix is therefore loaded from HBM exactly
once per step.

Simplifications in this baseline:
* B>1 prompts are processed sequentially (one fused buffer per prompt).
* Garbage collection is mask-only — eliminated beams' suffix slots are not
  physically reclaimed. The buffer is preallocated to ``L_p + K * max_new``
  per prompt. (The compaction variant would copy live slots to compact the
  buffer; we leave that as future work since the algorithmic structure is the
  point of comparison.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

from ..models.attention import AttentionContext


# ---------------------------------------------------------------------------
# Attention context — writes k/v into a fused buffer, runs ragged prefill
# ---------------------------------------------------------------------------


@dataclass
class TreeAttentionContext(AttentionContext):
    """Per-forward state for tree attention over a single prompt's fused buffer."""

    kv_cache: list[torch.Tensor]      # per layer: [max_tokens, 2, num_kv_heads, head_dim]
    write_indices: torch.Tensor       # [nnz] int64 — buffer index per input token to write
    kv_total_len: int                 # number of valid KV entries after this forward writes
    num_kv_heads: int
    head_dim: int
    num_qo_heads: int
    wrapper: BatchPrefillWithRaggedKVCacheWrapper

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        kv = self.kv_cache[layer_idx]

        k_3d = k.view(-1, self.num_kv_heads, self.head_dim)
        v_3d = v.view(-1, self.num_kv_heads, self.head_dim)

        # Append k/v into the fused buffer at the requested slots.
        kv[self.write_indices, 0] = k_3d
        kv[self.write_indices, 1] = v_3d

        # Slice the active region as the ragged K/V tensor for FlashAttention.
        k_view = kv[: self.kv_total_len, 0]   # [total_len, num_kv_heads, head_dim]
        v_view = kv[: self.kv_total_len, 1]

        q_3d = q.view(-1, self.num_qo_heads, self.head_dim)
        out = self.wrapper.run(q_3d, k_view, v_view)
        return out.reshape(*q.shape[:-1], self.num_qo_heads * self.head_dim)


# ---------------------------------------------------------------------------
# Beam state
# ---------------------------------------------------------------------------


@dataclass
class Beam:
    token_ids: list[int]      # generated tokens (excluding prompt)
    cum_log_prob: float
    path: list[int]           # ordered fused-buffer indices on this beam's KV path


# ---------------------------------------------------------------------------
# Per-prompt beam search
# ---------------------------------------------------------------------------


def _build_tree_mask(beams: list[Beam], kv_total_len: int, device) -> torch.Tensor:
    """Bool mask [K, kv_total_len]: True where beam k's query may attend to key j."""
    K = len(beams)
    mask = torch.zeros((K, kv_total_len), dtype=torch.bool, device=device)
    for k, beam in enumerate(beams):
        idx = torch.tensor(beam.path, dtype=torch.long, device=device)
        mask[k, idx] = True
    return mask


def _beam_search_single(
    model,
    config,
    prompt_ids: list[int],
    max_new_tokens: int,
    beam_width: int,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    timings: dict | None = None,
) -> list[Beam]:
    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers

    K = beam_width
    L_p = len(prompt_ids)
    max_tokens = L_p + K * max_new_tokens + K  # +K head room

    # Per-layer fused KV buffer.
    kv_cache = [
        torch.zeros(
            (max_tokens, 2, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer, kv_layout="NHD")

    # ----- Prefill: a single causal request of length L_p -----
    prefill_write_indices = torch.arange(L_p, dtype=torch.long, device=device)

    qo_indptr = torch.tensor([0, L_p], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, L_p], dtype=torch.int32, device=device)
    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        causal=True,
    )

    ctx = TreeAttentionContext(
        kv_cache=kv_cache,
        write_indices=prefill_write_indices,
        kv_total_len=L_p,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_qo_heads=num_qo_heads,
        wrapper=wrapper,
    )

    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    positions = torch.arange(L_p, device=device).unsqueeze(0)

    if timings is not None:
        torch.cuda.synchronize()
        t_pre = time.perf_counter()

    with torch.no_grad():
        hidden_states = model.forward(input_ids=input_ids, positions=positions, ctx=ctx)
        logits = model.compute_logits(hidden_states[:, -1, :])
        log_probs = F.log_softmax(logits, dim=-1)
        topk_log_probs, topk_ids = log_probs.topk(K, dim=-1)
        topk_log_probs = topk_log_probs.squeeze(0)
        topk_ids = topk_ids.squeeze(0)

    if timings is not None:
        torch.cuda.synchronize()
        timings["prefill_ms"] = (time.perf_counter() - t_pre) * 1000.0

    # All K beams initially share the prefix path [0, L_p).
    prefix_path = list(range(L_p))
    beams: list[Beam] = [
        Beam(
            token_ids=[topk_ids[i].item()],
            cum_log_prob=topk_log_probs[i].item(),
            path=list(prefix_path),
        )
        for i in range(K)
    ]
    kv_total_len = L_p

    # ----- Decode loop -----
    current_pos = L_p

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if timings is not None:
                torch.cuda.synchronize()
                t_step = time.perf_counter()
            # 1) Allocate K new slots, one per beam (slot k holds beam k's KV for this step).
            new_slots = list(range(kv_total_len, kv_total_len + K))
            kv_total_len += K
            for k in range(K):
                beams[k].path.append(new_slots[k])

            # 2) Build tree mask [K, kv_total_len].
            mask = _build_tree_mask(beams, kv_total_len, device)

            # 3) Plan ragged prefill: 1 request, qo_len=K, kv_len=kv_total_len, custom mask.
            qo_indptr = torch.tensor([0, K], dtype=torch.int32, device=device)
            kv_indptr = torch.tensor([0, kv_total_len], dtype=torch.int32, device=device)
            wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                causal=False,
                custom_mask=mask.flatten(),
            )

            ctx = TreeAttentionContext(
                kv_cache=kv_cache,
                write_indices=torch.tensor(new_slots, dtype=torch.long, device=device),
                kv_total_len=kv_total_len,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_qo_heads=num_qo_heads,
                wrapper=wrapper,
            )

            beam_input_ids = torch.tensor(
                [[beam.token_ids[-1]] for beam in beams],
                dtype=torch.long, device=device,
            )  # [K, 1]
            beam_positions = torch.tensor(
                [[current_pos] for _ in range(K)], device=device,
            )  # [K, 1]

            hidden_states = model.forward(
                input_ids=beam_input_ids,
                positions=beam_positions,
                ctx=ctx,
            )
            logits = model.compute_logits(hidden_states[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)

            # 4) Score and pick top-K.
            vocab_size = logits.shape[-1]
            cum_probs = torch.tensor(
                [beam.cum_log_prob for beam in beams],
                device=device, dtype=torch.float32,
            )
            scores = cum_probs[:, None] + log_probs.float()
            flat_scores = scores.reshape(-1)
            topk_scores, topk_flat_ids = flat_scores.topk(K)

            parent_beam_ids = (topk_flat_ids // vocab_size).tolist()
            new_token_ids = (topk_flat_ids % vocab_size).tolist()

            # 5) Build new beam list. Forking is fine — multiple new beams may share a parent
            #    path, but each will diverge by appending its own new slot at the next step.
            new_beams: list[Beam] = []
            for i in range(K):
                pid = parent_beam_ids[i]
                new_beams.append(
                    Beam(
                        token_ids=beams[pid].token_ids + [new_token_ids[i]],
                        cum_log_prob=topk_scores[i].item(),
                        path=list(beams[pid].path),
                    )
                )
            beams = new_beams
            current_pos += 1

            if timings is not None:
                torch.cuda.synchronize()
                timings["decode_step_ms"].append((time.perf_counter() - t_step) * 1000.0)

    beams.sort(key=lambda beam: beam.cum_log_prob, reverse=True)
    return beams


def beam_search(
    model,
    config,
    prompt_ids: list[list[int]],
    max_new_tokens: int,
    beam_width: int,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    return_timings: bool = False,
):
    """Tree-attention beam search.

    Multi-prompt batches run sequentially (one fused buffer per prompt). For
    a fair single-prompt comparison against paged-attention, this is equivalent
    to B=1. When ``return_timings=True``, returns ``(beams, timings)`` where
    timings aggregates over all prompts (per-step lists are concatenated).
    """
    if return_timings:
        agg = {"prefill_ms": 0.0, "decode_step_ms": []}
        all_beams: list[list[Beam]] = []
        for p in prompt_ids:
            t = {"prefill_ms": 0.0, "decode_step_ms": []}
            beams = _beam_search_single(
                model, config, p, max_new_tokens, beam_width,
                device=device, dtype=dtype, timings=t,
            )
            all_beams.append(beams)
            agg["prefill_ms"] += t["prefill_ms"]
            agg["decode_step_ms"].extend(t["decode_step_ms"])
        return all_beams, agg
    return [
        _beam_search_single(
            model, config, p, max_new_tokens, beam_width,
            device=device, dtype=dtype,
        )
        for p in prompt_ids
    ]

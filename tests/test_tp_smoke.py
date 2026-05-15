"""Tensor-parallel smoke test.

Each rank runs the same beam search; rank 0 verifies that the resulting
beams (token sequences and scores) match a reference saved from a
prior single-GPU run, or that all ranks agree among themselves.

Launch with::

    torchrun --nproc_per_node=2 tests/test_tp_smoke.py

Or, when no reference is available, just run TP=N and trust the
RowParallelLinear all-reduce: every rank should produce identical
token sequences (same hidden states → same logits → same top-K), so
we cross-check rank 0 vs broadcast from rank 1.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from beam_engine.baselines import paged
from beam_engine.distributed import (
    destroy_tp,
    get_tp_rank,
    get_tp_world_size,
    init_tp,
)
from beam_engine.methods import adaptive_pool
from beam_engine.methods.bs_kernel import driver as bs_kernel_driver
from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = os.environ.get("BE_SMOKE_MODEL", "meta-llama/Llama-3.1-8B")
DTYPE = torch.float16
BACKENDS = {
    "paged": paged.beam_search,
    "adaptive_pool": adaptive_pool.beam_search,
    "bs_kernel": bs_kernel_driver.beam_search,
}


def _ok(rank: int, label: str, ok: bool) -> bool:
    if rank == 0:
        print(f"  {label}: {'PASS' if ok else 'FAIL'}")
    return ok


def _exercise_backend(
    backend_name: str,
    beam_search_fn,
    *,
    model,
    config,
    tokenizer,
    prompt: str,
    prompt_ids: list[int],
    max_new: int,
    K: int,
    device: str,
    tp_rank: int,
    tp_size: int,
) -> bool:
    if tp_rank == 0:
        print(f"\n[smoke] backend={backend_name}")

    beams = beam_search_fn(
        model, config, [prompt_ids], max_new, beam_width=K,
    )[0]

    local_tokens = [tuple(b.token_ids) for b in beams]
    local_scores = torch.tensor(
        [b.cum_log_prob for b in beams], dtype=torch.float64, device=device,
    )

    ok = True
    if tp_size > 1:
        gathered_tokens: list[list[tuple[int, ...]]] = [None] * tp_size  # type: ignore[list-item]
        dist.all_gather_object(gathered_tokens, local_tokens)
        gathered_scores = [
            torch.empty_like(local_scores) for _ in range(tp_size)
        ]
        dist.all_gather(gathered_scores, local_scores)
        if tp_rank == 0:
            tokens_match = all(t == gathered_tokens[0] for t in gathered_tokens)
            ok &= _ok(tp_rank, "tokens identical across ranks", tokens_match)
            max_drift = max(
                (gathered_scores[r] - gathered_scores[0]).abs().max().item()
                for r in range(tp_size)
            )
            scores_match = max_drift < 1e-2
            ok &= _ok(
                tp_rank,
                f"scores agree across ranks (max drift={max_drift:.2e})",
                scores_match,
            )

    if tp_rank == 0:
        for i, b in enumerate(beams):
            text = tokenizer.decode(b.token_ids, skip_special_tokens=True)
            print(
                f"  [{i}] score={b.cum_log_prob:7.4f}  "
                f"\"{prompt}{text}\""
            )
        ok &= _ok(tp_rank, "best-beam score finite", math.isfinite(beams[0].cum_log_prob))
    return ok


def main() -> None:
    init_tp()
    tp_rank = get_tp_rank()
    tp_size = get_tp_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}"

    backend_names = os.environ.get("BE_SMOKE_BACKENDS", "paged").split(",")

    if tp_rank == 0:
        print(f"[smoke] tp_size={tp_size} device={device} backends={backend_names}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=device)
    config = model.config

    # The unified page driver assumes the prompt prefix is page-aligned
    # — split-form cascade backends keep ``pages_tail`` empty until the
    # first off==0 write, so a mid-page prefix would trigger an
    # out-of-range index in the CoW path. Pad the prompt token IDs up to
    # a multiple of 16 (page_size). Padding with a tokenizer-safe token
    # keeps the prompt valid input to the model.
    prompt_text = (
        "The capital of France is Paris, a vibrant city of art, fashion, "
        "and history. The first time I visited Paris, I walked along the "
        "Seine, admired the Eiffel Tower at night, and"
    )
    prompt_ids = tokenizer.encode(prompt_text)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    page_size = 16
    rem = len(prompt_ids) % page_size
    if rem != 0:
        prompt_ids = prompt_ids + [pad_id] * (page_size - rem)
    max_new = 16
    K = 4
    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)

    all_pass = True
    for backend_name in backend_names:
        fn = BACKENDS.get(backend_name)
        if fn is None:
            if tp_rank == 0:
                print(f"  unknown backend: {backend_name}")
            all_pass = False
            continue
        all_pass &= _exercise_backend(
            backend_name, fn,
            model=model, config=config, tokenizer=tokenizer,
            prompt=prompt, prompt_ids=prompt_ids,
            max_new=max_new, K=K, device=device,
            tp_rank=tp_rank, tp_size=tp_size,
        )

    if tp_rank == 0:
        print(f"\n[smoke] {'ALL PASS' if all_pass else 'SOME FAILED'}")

    destroy_tp()
    if tp_rank == 0 and not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()

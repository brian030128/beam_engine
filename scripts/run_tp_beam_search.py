"""Multi-GPU beam-search runner.

Each rank loads its TP shard of the model, then every rank runs the
same ``beam_search`` call. CPU-side beam / page state is computed
identically across ranks (same inputs, deterministic logic), and the
all-reduces inside ``RowParallelLinear`` keep activations in sync — so
each rank's top-K selects the same beams.

Launch with::

    torchrun --nproc_per_node=$TP scripts/run_tp_beam_search.py \\
        --model meta-llama/Llama-3.1-8B \\
        --backend paged \\
        --beam-width 4 \\
        --max-new 20

Only rank 0 prints the decoded beams.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

from beam_engine.distributed import (
    destroy_tp,
    get_tp_rank,
    get_tp_world_size,
    init_tp,
)
from beam_engine.models.modeling_llama import LlamaForCausalLM


_BACKENDS = {
    "paged": ("beam_engine.baselines.paged", "beam_search"),
    "tree": ("beam_engine.baselines.tree", "beam_search"),
    "mlca": ("beam_engine.baselines.mlca", "beam_search"),
    "fasttree": ("beam_engine.baselines.fasttree", "beam_search"),
    "adaptive_pool": ("beam_engine.methods.adaptive_pool", "beam_search"),
}


def _load_backend(name: str):
    mod_path, fn = _BACKENDS[name]
    import importlib

    return getattr(importlib.import_module(mod_path), fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="TP beam-search runner")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--backend",
        choices=sorted(_BACKENDS),
        default="paged",
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=20)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    args = parser.parse_args()

    init_tp()  # picks up RANK/WORLD_SIZE/LOCAL_RANK from torchrun env

    tp_rank = get_tp_rank()
    tp_size = get_tp_world_size()
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if tp_rank == 0:
        print(f"TP world_size={tp_size}, loading {args.model} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(args.model, dtype=dtype, device=device)
    config = model.config

    prompt_ids = tokenizer.encode(args.prompt)
    beam_search = _load_backend(args.backend)
    beams = beam_search(
        model,
        config,
        [prompt_ids],
        args.max_new,
        beam_width=args.beam_width,
    )[0]

    if tp_rank == 0:
        print(f"\nBackend={args.backend}  prompt={args.prompt!r}")
        for i, b in enumerate(beams):
            text = tokenizer.decode(b.token_ids, skip_special_tokens=True)
            print(
                f"  [{i}] score={b.cum_log_prob:7.4f}  "
                f"\"{args.prompt}{text}\""
            )

    destroy_tp()


if __name__ == "__main__":
    main()

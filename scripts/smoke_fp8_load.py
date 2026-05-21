"""Minimal fp8 load + forward smoke.

Loads the fp8 70B (or any fp8 model passed via --model) under TP, runs
one short beam_search through ``paged`` and ``bs_kernel``, and verifies
both backends produce token-identical outputs. No fixture: this hits the
real HF weights so any loader bug surfaces here, not at job-cell time.

Launch with::

    torchrun --nproc_per_node=2 scripts/smoke_fp8_load.py \\
        --model RedHatAI/Meta-Llama-3-70B-Instruct-FP8 --max-new 8
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoTokenizer

from beam_engine.baselines.paged import beam_search as paged_beam_search
from beam_engine.distributed import destroy_tp, get_tp_rank, init_tp
from beam_engine.methods.bs_kernel import beam_search as bsk_beam_search
from beam_engine.models import load_model_for_causal_lm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--max-new", type=int, default=8)
    p.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    p.add_argument("--kv-dtype", choices=("bf16", "fp16", "fp8_e4m3"), default=None,
                   help="KV cache storage dtype (default: same as --dtype)")
    args = p.parse_args()

    init_tp()
    rank = get_tp_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    kv_dtype_map = {
        None: None, "bf16": torch.bfloat16, "fp16": torch.float16,
        "fp8_e4m3": torch.float8_e4m3fn,
    }
    kv_dtype = kv_dtype_map[args.kv_dtype]

    def say(*a, **kw):
        if rank == 0:
            print(*a, **kw, flush=True)

    say(f"loading {args.model} on {device} (dtype={args.dtype}, kv_dtype={args.kv_dtype or args.dtype})...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_model_for_causal_lm(args.model, dtype=dtype, device=device)
    config = model.config
    say(f"loaded; cuda.memory_allocated={torch.cuda.memory_allocated(device) / 2**30:.2f} GiB / rank")

    prompt_ids = tok.encode(args.prompt)
    say(f"prompt len={len(prompt_ids)}, beam_width={args.beam_width}, max_new={args.max_new}")

    kw = {"dtype": dtype}
    if kv_dtype is not None:
        kw["kv_dtype"] = kv_dtype

    say("running paged beam_search...")
    paged_beams = paged_beam_search(
        model, config, [prompt_ids], args.max_new, beam_width=args.beam_width,
        **kw,
    )[0]

    say("running bs_kernel beam_search...")
    bsk_beams = bsk_beam_search(
        model, config, [prompt_ids], args.max_new, beam_width=args.beam_width,
        **kw,
    )[0]

    if rank == 0:
        ok = True
        for i, (pb, bb) in enumerate(zip(paged_beams, bsk_beams)):
            same = pb.token_ids == bb.token_ids
            ok = ok and same
            print(f"  beam[{i}] paged={pb.token_ids} bsk={bb.token_ids} match={same}")
            print(f"           paged.score={pb.cum_log_prob:.4f} bsk.score={bb.cum_log_prob:.4f}")
        print("PASS" if ok else "FAIL: paged vs bs_kernel diverged")
    destroy_tp()


if __name__ == "__main__":
    main()

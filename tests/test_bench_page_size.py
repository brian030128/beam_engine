"""
Benchmark beam search across different page sizes.

Measures latency for beam search (beam_width=4, max_new_tokens=20)
with page sizes [1, 2, 4, 8, 16, 32, 64] and prints a summary table.
"""

import time

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from transformers import AutoTokenizer

from beam_engine.models.modeling_llama import LlamaForCausalLM
from beam_engine.page_table import PageTable
from test_beam_search import beam_search

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16

PAGE_SIZES = [1, 2, 4, 8, 16, 32, 64]
BEAM_WIDTH = 4
MAX_NEW_TOKENS = 20
WARMUP_RUNS = 2
TIMED_RUNS = 5

PROMPT = "The capital of France is"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded!\n")

    prompt_ids = tokenizer.encode(PROMPT)

    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers

    results = []

    for ps in PAGE_SIZES:
        print(f"--- page_size={ps} ---")

        # Create reusable resources for this page size
        page_table = PageTable(
            layer_num=num_layers,
            page_size=ps,
            max_num_pages=2048,
            head_num=num_kv_heads,
            head_dim=head_dim,
            device=torch.device(DEVICE),
            store_dtype=DTYPE,
        )
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
        prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")
        decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout="NHD", use_tensor_cores=True,
        )
        reuse_kwargs = dict(
            page_table=page_table,
            workspace_buffer=workspace_buffer,
            prefill_wrapper=prefill_wrapper,
            decode_wrapper=decode_wrapper,
            page_size=ps,
        )

        # Warmup
        for _ in range(WARMUP_RUNS):
            beam_search(
                model, config, [prompt_ids], MAX_NEW_TOKENS, BEAM_WIDTH,
                **reuse_kwargs,
            )

        # Timed runs
        latencies: list[float] = []
        for i in range(TIMED_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            beams = beam_search(
                model, config, [prompt_ids], MAX_NEW_TOKENS, BEAM_WIDTH,
                **reuse_kwargs,
            )
            torch.cuda.synchronize()
            t = time.perf_counter() - t0
            latencies.append(t)
            print(f"  iter {i + 1}/{TIMED_RUNS}: {t * 1e3:.1f} ms")

        avg_ms = sum(latencies) / len(latencies) * 1e3

        best_beam = beams[0][0]
        text = tokenizer.decode(best_beam.token_ids, skip_special_tokens=True)
        results.append((ps, avg_ms, text))
        print(f"  avg={avg_ms:.1f}ms  text=\"{PROMPT}{text}\"\n")

    # Summary table
    print("=" * 70)
    print(f"{'page_size':>10}  {'avg_ms':>10}  {'output'}")
    print("-" * 70)
    for ps, avg_ms, text in results:
        print(f"{ps:>10}  {avg_ms:>10.1f}  {PROMPT}{text}")
    print("=" * 70)


if __name__ == "__main__":
    main()

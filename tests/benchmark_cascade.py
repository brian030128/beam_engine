"""
Benchmark cascade vs flat attention for beam search.

Compares MultiLevelCascadeAttentionWrapper (shared prefix optimization)
against BatchDecodeWithPagedKVCacheWrapper (flat, all pages per beam).
"""

import time

import numpy as np
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
)
from transformers import AutoTokenizer

from test_beam_search import beam_search

from beam_engine.models.modeling_llama import LlamaForCausalLM
from beam_engine.page_table import PageTable

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16
PAGE_SIZE = 4

BATCH_SIZE = 64
PROMPT_LEN = 2048
OUTPUT_LEN = 20
BEAM_WIDTH = 4

NUM_WARMUP = 3
NUM_ITERS = 10


def _make_prompts(rng: np.random.Generator) -> list[list[int]]:
    return [
        rng.integers(1000, 30000, size=PROMPT_LEN).tolist()
        for _ in range(BATCH_SIZE)
    ]


def _run_once(
    model, config, prompts, use_cascade, *,
    page_table, workspace_buffer, prefill_wrapper, decode_wrapper,
):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    beam_search(
        model, config, prompts, OUTPUT_LEN, BEAM_WIDTH,
        use_cascade=use_cascade,
        page_table=page_table,
        workspace_buffer=workspace_buffer,
        prefill_wrapper=prefill_wrapper,
        decode_wrapper=decode_wrapper,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench(model, config, prompts, use_cascade: bool, label: str) -> float:
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers

    # Estimate max pages: each prompt needs ceil(2048/4)=512 pages,
    # plus each beam can grow up to ceil(20/4)=5 extra pages.
    # Total worst case: 64 * 4 * (512 + 5) = ~132k pages. Use a generous budget.
    max_num_pages = BATCH_SIZE * BEAM_WIDTH * ((PROMPT_LEN + OUTPUT_LEN + PAGE_SIZE - 1) // PAGE_SIZE) + 1024
    page_table = PageTable(
        layer_num=num_layers,
        page_size=PAGE_SIZE,
        max_num_pages=max_num_pages,
        head_num=num_kv_heads,
        head_dim=head_dim,
        device=torch.device(DEVICE),
        store_dtype=DTYPE,
    )
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")
    if use_cascade:
        decode_wrapper = MultiLevelCascadeAttentionWrapper(
            num_levels=2, float_workspace_buffer=workspace_buffer, kv_layout="NHD",
        )
    else:
        decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout="NHD", use_tensor_cores=True,
        )

    kwargs = dict(
        page_table=page_table,
        workspace_buffer=workspace_buffer,
        prefill_wrapper=prefill_wrapper,
        decode_wrapper=decode_wrapper,
    )

    print(f"[{label}] Warming up ({NUM_WARMUP} iters)...")
    for _ in range(NUM_WARMUP):
        _run_once(model, config, prompts, use_cascade, **kwargs)

    print(f"[{label}] Benchmarking ({NUM_ITERS} iters)...")
    latencies: list[float] = []
    for i in range(NUM_ITERS):
        t = _run_once(model, config, prompts, use_cascade, **kwargs)
        latencies.append(t)
        print(f"  iter {i + 1:2d}/{NUM_ITERS}: {t * 1e3:.1f} ms")

    avg_ms = np.mean(latencies) * 1e3
    std_ms = np.std(latencies) * 1e3
    print(f"[{label}] avg = {avg_ms:.1f} ms  (std = {std_ms:.1f} ms)\n")

    del page_table, workspace_buffer, prefill_wrapper, decode_wrapper
    torch.cuda.empty_cache()

    return avg_ms


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    prompts = _make_prompts(rng)

    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded.\n")

    print("=" * 65)
    print(f"  Cascade vs Flat attention — beam search decode")
    print(f"  Model:        {MODEL_NAME}")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Beam width:   {BEAM_WIDTH}")
    print(f"  Prompt len:   {PROMPT_LEN} tokens")
    print(f"  Output len:   {OUTPUT_LEN} tokens")
    print(f"  Page size:    {PAGE_SIZE}")
    print(f"  Warmup:       {NUM_WARMUP}  |  Bench iters: {NUM_ITERS}")
    print("=" * 65)
    print()

    cascade_ms = bench(model, config, prompts, use_cascade=True, label="cascade")
    flat_ms = bench(model, config, prompts, use_cascade=False, label="flat")

    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Cascade:  {cascade_ms:8.1f} ms")
    print(f"  Flat:     {flat_ms:8.1f} ms")
    speedup = flat_ms / cascade_ms
    print(f"  Speedup:  {speedup:.2f}x  ({'cascade wins' if speedup > 1 else 'flat wins'})")
    print("=" * 65)

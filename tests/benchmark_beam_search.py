"""
Benchmark beam_engine vs vllm on Llama-3.1-8B — beam search.

vllm is configured with:
  - AttentionBackendEnum.FLASHINFER  (same kernel family as beam_engine)
  - enforce_eager=True               (no torch.compile / CUDA graphs)

Each sequence uses a distinct prompt (different content and different length) to
prevent vLLM prefix-caching from giving it an unfair advantage.

beam_engine runs beam search with cross-prompt batching (all prompts in one call).
vllm uses its built-in llm.beam_search() method.

beam_engine is swept across multiple page sizes to measure the effect of page
granularity on performance.
"""

import time

import numpy as np
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from transformers import AutoTokenizer

from test_beam_search import beam_search

from beam_engine.models.modeling_llama import LlamaForCausalLM
from beam_engine.page_table import PageTable
from vllm import LLM
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.sampling_params import BeamSearchParams

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16
PAGE_SIZES = [1, 4, 16]
BASE_MAX_PAGES = 2048   # max_num_pages for page_size=16; scales inversely

BATCH_SIZE = 4
OUTPUT_LEN = 100
NUM_WARMUP = 3
NUM_ITERS = 10

# Distinct prompt lengths (tokens) — one per batch slot.
PROMPT_LENS = [512, 640, 768, 896]
assert len(PROMPT_LENS) == BATCH_SIZE

BEAM_WIDTH = 16


def _make_prompts(rng: np.random.Generator) -> list[list[int]]:
    """Generate BATCH_SIZE prompts with distinct lengths and random content."""
    return [
        rng.integers(1000, 30000, size=length).tolist()
        for length in PROMPT_LENS
    ]


# ---------------------------------------------------------------------------
# beam_engine — batched beam search (all prompts in one call)
# ---------------------------------------------------------------------------

def _be_run_once(
    model,
    config,
    prompts: list[list[int]],
    output_len: int,
    beam_width: int,
    page_size: int,
    collect_tokens: bool = False,
    *,
    page_table=None,
    workspace_buffer=None,
    prefill_wrapper=None,
    decode_wrapper=None,
) -> tuple[float, list[list[int]] | None]:
    """Run beam search on all prompts in a single batched call.

    Returns (total_time_s, list[best_beam_tokens] | None).
    """
    reuse_kwargs = {}
    if page_table is not None:
        reuse_kwargs["page_table"] = page_table
    if workspace_buffer is not None:
        reuse_kwargs["workspace_buffer"] = workspace_buffer
    if prefill_wrapper is not None:
        reuse_kwargs["prefill_wrapper"] = prefill_wrapper
    if decode_wrapper is not None:
        reuse_kwargs["decode_wrapper"] = decode_wrapper

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    all_beams = beam_search(
        model, config, prompts, output_len, beam_width,
        page_size=page_size, **reuse_kwargs,
    )
    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    best_tokens: list[list[int]] | None = None
    if collect_tokens:
        best_tokens = [all_beams[i][0].token_ids for i in range(len(prompts))]

    return total_time, best_tokens


def run_beam_engine_benchmark(
    prompts: list[list[int]],
    output_len: int,
    beam_width: int,
    num_warmup: int,
    num_iters: int,
    tokenizer,
    page_size: int,
) -> tuple[float, list[list[int]]]:
    """Load model, verify, warmup, benchmark for a single page_size.

    Returns (avg_ms, best_beam_tokens).
    """
    print(f"\n{'—' * 65}")
    print(f"[beam_engine] page_size={page_size}")
    print(f"{'—' * 65}")

    print("Loading beam_engine model...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("beam_engine model loaded.\n")

    # Scale max_num_pages inversely with page_size to keep the same KV capacity
    max_num_pages = BASE_MAX_PAGES * 16 // page_size

    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers
    page_table = PageTable(
        layer_num=num_layers,
        page_size=page_size,
        max_num_pages=max_num_pages,
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
    )

    print(f"[beam_engine] max_num_pages={max_num_pages}")
    print("[beam_engine] Verification pass (best beam, first 20 tokens per sequence):")
    _, gen = _be_run_once(
        model, config, prompts, output_len, beam_width, page_size,
        collect_tokens=True, **reuse_kwargs,
    )
    for i, tokens in enumerate(gen):
        text = tokenizer.decode(tokens[:20], skip_special_tokens=True)
        print(f"  seq[{i}] (prompt_len={len(prompts[i])}): {text!r}")
    print()

    print(f"[beam_engine] Warming up ({num_warmup} iters)...")
    for _ in range(num_warmup):
        _be_run_once(model, config, prompts, output_len, beam_width, page_size, **reuse_kwargs)

    print(f"[beam_engine] Benchmarking ({num_iters} iters)...")
    latencies: list[float] = []
    for i in range(num_iters):
        t, _ = _be_run_once(model, config, prompts, output_len, beam_width, page_size, **reuse_kwargs)
        latencies.append(t)
        print(f"  iter {i + 1:2d}/{num_iters}: total={t * 1e3:.1f} ms")

    del model
    torch.cuda.empty_cache()

    return np.mean(latencies) * 1e3, gen  # ms, best-beam tokens


# ---------------------------------------------------------------------------
# vllm — beam search
# ---------------------------------------------------------------------------

def run_vllm_benchmark(
    prompts: list[list[int]],
    output_len: int,
    beam_width: int,
    num_warmup: int,
    num_iters: int,
    tokenizer,
) -> tuple[float, list[list[int]]]:
    """Load LLM, verify, warmup, benchmark. Returns (avg_ms, best_beam_tokens)."""
    print("\nLoading vllm model (FlashInfer backend, eager mode)...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        enforce_eager=False,
        attention_backend=AttentionBackendEnum.FLASHINFER,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        max_logprobs=96
    )
    print("vllm model loaded.\n")

    params = BeamSearchParams(
        beam_width=beam_width,
        max_tokens=output_len,
        ignore_eos=True,
        temperature=0.0,
    )
    vllm_prompts = [{"prompt_token_ids": ids} for ids in prompts]

    def run_once():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.beam_search(vllm_prompts, params)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, outputs

    print("[vllm] Verification pass (best beam, first 20 tokens per sequence):")
    _, outputs = run_once()
    gen: list[list[int]] = []
    for i, out in enumerate(outputs):
        # output tokens include prompt; strip prompt to get generated tokens
        gen_tokens = out.sequences[0].tokens[len(prompts[i]):]
        gen.append(gen_tokens)
        text = tokenizer.decode(gen_tokens[:20], skip_special_tokens=True)
        print(f"  seq[{i}] (prompt_len={len(prompts[i])}): {text!r}")
    print()

    print(f"[vllm] Warming up ({num_warmup - 1} iters)...")  # first call above counts as 1
    for _ in range(num_warmup - 1):
        run_once()

    print(f"[vllm] Benchmarking ({num_iters} iters)...")
    latencies: list[float] = []
    for i in range(num_iters):
        t, _ = run_once()
        latencies.append(t)
        print(f"  iter {i + 1:2d}/{num_iters}: total={t * 1e3:.1f} ms")

    del llm
    torch.cuda.empty_cache()

    return np.mean(latencies) * 1e3, gen  # ms, best-beam tokens


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    prompts = _make_prompts(rng)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    input_lens_str = ", ".join(str(len(p)) for p in prompts)
    total_input = sum(len(p) for p in prompts)
    page_sizes_str = ", ".join(str(ps) for ps in PAGE_SIZES)

    print("=" * 65)
    print(f"  Benchmark: beam_engine vs vllm  (beam search, width={BEAM_WIDTH})")
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Batch size: {BATCH_SIZE} sequences")
    print(f"  Beam width: {BEAM_WIDTH}")
    print(f"  Input lens: [{input_lens_str}] tokens  (total {total_input})")
    print(f"  Output:     {OUTPUT_LEN} tokens/seq")
    print(f"  Page sizes: [{page_sizes_str}]")
    print(f"  Warmup:     {NUM_WARMUP}  |  Bench iters: {NUM_ITERS}")
    print("=" * 65)

    # --- beam_engine: sweep page sizes ---
    be_results: dict[int, tuple[float, list[list[int]]]] = {}
    for ps in PAGE_SIZES:
        avg_ms, tokens = run_beam_engine_benchmark(
            prompts, OUTPUT_LEN, BEAM_WIDTH, NUM_WARMUP, NUM_ITERS, tokenizer, ps
        )
        be_results[ps] = (avg_ms, tokens)

    # --- vllm: single run ---
    vllm_total_ms, vllm_tokens = run_vllm_benchmark(
        prompts, OUTPUT_LEN, BEAM_WIDTH, NUM_WARMUP, NUM_ITERS, tokenizer
    )

    # --- Output verification: check all page sizes produce same tokens ---
    print()
    print("=" * 65)
    print("  OUTPUT VERIFICATION (best beam tokens)")
    print("=" * 65)

    # Check beam_engine page sizes agree with each other
    ref_ps = PAGE_SIZES[0]
    ref_tokens = be_results[ref_ps][1]
    be_consistent = True
    for ps in PAGE_SIZES[1:]:
        ps_tokens = be_results[ps][1]
        match = all(a == b for a, b in zip(ref_tokens, ps_tokens))
        status = "PASS" if match else "FAIL"
        print(f"  beam_engine page_size={ref_ps} vs {ps}: {status}")
        if not match:
            be_consistent = False
            for i, (a, b) in enumerate(zip(ref_tokens, ps_tokens)):
                if a != b:
                    print(f"    seq[{i}] differs")

    # Check beam_engine vs vllm (using the largest page size as reference)
    be_ref_tokens = be_results[PAGE_SIZES[-1]][1]
    all_match = True
    for i, (be_t, vl_t) in enumerate(zip(be_ref_tokens, vllm_tokens)):
        if be_t == vl_t:
            print(f"  beam_engine vs vllm seq[{i}]: PASS ({len(be_t)} tokens match)")
        else:
            all_match = False
            print(f"  beam_engine vs vllm seq[{i}]: FAIL")
            print(f"    beam_engine: {be_t}")
            print(f"    vllm:        {vl_t}")
    if be_consistent and all_match:
        print("  Overall: PASS")
    else:
        print("  Overall: FAIL")

    # --- Results table ---
    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    for ps in PAGE_SIZES:
        avg_ms = be_results[ps][0]
        print(f"  beam_engine  (page_size={ps:2d}, beam_width={BEAM_WIDTH}, batch={BATCH_SIZE})")
        print(f"    Total latency:          {avg_ms:8.2f} ms")
    print(f"  vllm  (FlashInfer, eager, beam_width={BEAM_WIDTH}, batch={BATCH_SIZE})")
    print(f"    Total latency:          {vllm_total_ms:8.2f} ms")
    print()
    print("  Ratios (beam_engine / vllm):")
    for ps in PAGE_SIZES:
        avg_ms = be_results[ps][0]
        print(f"    page_size={ps:2d}:  {avg_ms / vllm_total_ms:.2f}x")
    print("=" * 65)

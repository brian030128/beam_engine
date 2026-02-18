"""
Benchmark beam_engine vs vllm on Llama-3.1-8B.

vllm is configured with:
  - AttentionBackendEnum.FLASHINFER  (same kernel family as beam_engine)
  - enforce_eager=True               (no torch.compile / CUDA graphs)

beam_engine reports prefill and per-token decode latency separately.
vllm reports total generation latency (prefill + all decode steps).
"""

import time

import numpy as np
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)

from beam_engine.models.attention import AttentionMetadata
from beam_engine.models.modeling_llama import LlamaForCausalLM
from beam_engine.page_table import PageTable
from vllm import LLM, SamplingParams
from vllm.attention.backends.registry import AttentionBackendEnum

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16
PAGE_SIZE = 16

INPUT_LEN = 128
OUTPUT_LEN = 512
NUM_WARMUP = 3
NUM_ITERS = 10


# ---------------------------------------------------------------------------
# beam_engine
# ---------------------------------------------------------------------------

def _be_run_once(model, config, prompt_token_ids: list[int], output_len: int):
    """One full prefill + decode run. Returns (prefill_s, list[decode_s])."""
    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers

    page_table = PageTable(
        layer_num=num_layers,
        page_size=PAGE_SIZE,
        max_num_pages=1024,
        head_num=num_kv_heads,
        head_dim=head_dim,
        device=torch.device(DEVICE),
        store_dtype=DTYPE,
    )
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")
    decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD")

    prompt_len = len(prompt_token_ids)
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=DEVICE)

    # Allocate pages for the prompt
    pages: list[int] = []
    num_prompt_pages = (prompt_len + PAGE_SIZE - 1) // PAGE_SIZE
    for _ in range(num_prompt_pages):
        pages.append(page_table.allocate_block())

    token_positions = torch.arange(prompt_len, device=DEVICE)
    kv_page_indices = torch.tensor(
        [pages[i // PAGE_SIZE] for i in range(prompt_len)], dtype=torch.int32, device=DEVICE
    )
    kv_page_offsets = (token_positions % PAGE_SIZE).to(torch.int32)

    last_page_len = prompt_len - (num_prompt_pages - 1) * PAGE_SIZE
    prefill_wrapper.plan(
        qo_indptr=torch.tensor([0, prompt_len], dtype=torch.int32, device=DEVICE),
        paged_kv_indptr=torch.tensor([0, len(pages)], dtype=torch.int32, device=DEVICE),
        paged_kv_indices=torch.tensor(pages, dtype=torch.int32, device=DEVICE),
        paged_kv_last_page_len=torch.tensor([last_page_len], dtype=torch.int32, device=DEVICE),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=PAGE_SIZE,
        causal=True,
    )

    positions = torch.arange(prompt_len, device=DEVICE).unsqueeze(0)
    attn_metadata = AttentionMetadata(
        is_prefill=True,
        prefill_wrapper=prefill_wrapper,
        decode_wrapper=None,
        page_table=page_table,
        kv_page_indices=kv_page_indices,
        kv_page_offsets=kv_page_offsets,
    )

    # --- Prefill ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        hidden_states = model.forward(
            input_ids=input_ids, positions=positions, attn_metadata=attn_metadata
        )
        logits = model.compute_logits(hidden_states[:, -1:, :])
        next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0

    # --- Decode loop ---
    current_pos = prompt_len
    decode_times: list[float] = []

    with torch.no_grad():
        for _ in range(output_len - 1):
            if current_pos % PAGE_SIZE == 0:
                pages.append(page_table.allocate_block())

            page_idx = pages[current_pos // PAGE_SIZE]
            page_offset = current_pos % PAGE_SIZE
            decode_last_page_len = (current_pos % PAGE_SIZE) + 1

            decode_wrapper.plan(
                indptr=torch.tensor([0, len(pages)], dtype=torch.int32, device=DEVICE),
                indices=torch.tensor(pages, dtype=torch.int32, device=DEVICE),
                last_page_len=torch.tensor([decode_last_page_len], dtype=torch.int32, device=DEVICE),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=PAGE_SIZE,
            )

            attn_metadata = AttentionMetadata(
                is_prefill=False,
                prefill_wrapper=None,
                decode_wrapper=decode_wrapper,
                page_table=page_table,
                kv_page_indices=torch.tensor([page_idx], dtype=torch.int32, device=DEVICE),
                kv_page_offsets=torch.tensor([page_offset], dtype=torch.int32, device=DEVICE),
            )

            decode_positions = torch.tensor([[current_pos]], device=DEVICE)
            torch.cuda.synchronize()
            t_step = time.perf_counter()
            hidden_states = model.forward(
                input_ids=next_token_id,
                positions=decode_positions,
                attn_metadata=attn_metadata,
            )
            logits = model.compute_logits(hidden_states[:, -1:, :])
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - t_step)

            current_pos += 1

    return prefill_time, decode_times


def run_beam_engine_benchmark(prompt_token_ids: list[int], output_len: int,
                               num_warmup: int, num_iters: int):
    print("Loading beam_engine model...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("beam_engine model loaded.\n")

    print(f"[beam_engine] Warming up ({num_warmup} iters)...")
    for _ in range(num_warmup):
        _be_run_once(model, config, prompt_token_ids, output_len)

    print(f"[beam_engine] Benchmarking ({num_iters} iters)...")
    all_prefill: list[float] = []
    all_decode: list[float] = []
    for i in range(num_iters):
        p, d = _be_run_once(model, config, prompt_token_ids, output_len)
        all_prefill.append(p)
        all_decode.extend(d)
        print(f"  iter {i + 1:2d}/{num_iters}: prefill={p * 1e3:.1f} ms  "
              f"decode_avg={np.mean(d) * 1e3:.2f} ms/tok")

    del model
    torch.cuda.empty_cache()

    return np.mean(all_prefill) * 1e3, np.mean(all_decode) * 1e3  # ms


# ---------------------------------------------------------------------------
# vllm
# ---------------------------------------------------------------------------

def run_vllm_benchmark(prompt_token_ids: list[int], output_len: int,
                        num_warmup: int, num_iters: int):
    print("\nLoading vllm model (FlashInfer backend, eager mode)...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        enforce_eager=True,
        attention_backend=AttentionBackendEnum.FLASHINFER,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
    )
    print("vllm model loaded.\n")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_len,
        ignore_eos=True,
    )
    prompts = [{"prompt_token_ids": prompt_token_ids}]

    def run_once() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    print(f"[vllm] Warming up ({num_warmup} iters)...")
    for _ in range(num_warmup):
        run_once()

    print(f"[vllm] Benchmarking ({num_iters} iters)...")
    latencies: list[float] = []
    for i in range(num_iters):
        t = run_once()
        latencies.append(t)
        print(f"  iter {i + 1:2d}/{num_iters}: total={t * 1e3:.1f} ms")

    del llm
    torch.cuda.empty_cache()

    return np.mean(latencies) * 1e3  # ms


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    prompt_token_ids: list[int] = np.random.randint(1000, 30000, size=INPUT_LEN).tolist()

    print("=" * 65)
    print("  Benchmark: beam_engine vs vllm")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  Input:     {INPUT_LEN} tokens  |  Output: {OUTPUT_LEN} tokens")
    print(f"  Warmup:    {NUM_WARMUP}  |  Bench iters: {NUM_ITERS}")
    print("=" * 65)

    be_prefill_ms, be_decode_ms = run_beam_engine_benchmark(
        prompt_token_ids, OUTPUT_LEN, NUM_WARMUP, NUM_ITERS
    )

    vllm_total_ms = run_vllm_benchmark(
        prompt_token_ids, OUTPUT_LEN, NUM_WARMUP, NUM_ITERS
    )

    be_total_ms = be_prefill_ms + be_decode_ms * (OUTPUT_LEN - 1)

    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  beam_engine")
    print(f"    Prefill latency:        {be_prefill_ms:8.2f} ms")
    print(f"    Decode latency / token: {be_decode_ms:8.2f} ms")
    print(f"    Total ({INPUT_LEN} -> {INPUT_LEN + OUTPUT_LEN} tok):    {be_total_ms:8.2f} ms")
    print(f"  vllm  (FlashInfer, eager)")
    print(f"    Total latency:          {vllm_total_ms:8.2f} ms")
    print(f"  Ratio beam_engine / vllm: {be_total_ms / vllm_total_ms:8.2f}x")
    print("=" * 65)

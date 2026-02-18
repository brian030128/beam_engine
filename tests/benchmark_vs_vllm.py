"""
Benchmark beam_engine vs vllm on Llama-3.1-8B — batch decoding.

vllm is configured with:
  - AttentionBackendEnum.FLASHINFER  (same kernel family as beam_engine)
  - enforce_eager=True               (no torch.compile / CUDA graphs)

Each sequence uses a distinct prompt (different content and different length) to
prevent vLLM prefix-caching from giving it an unfair advantage.

beam_engine reports prefill and per-step decode latency separately.
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

BATCH_SIZE = 4
OUTPUT_LEN = 128
NUM_WARMUP = 3
NUM_ITERS = 10

# Distinct prompt lengths (tokens) — one per batch slot.
# Different lengths + different random content ensures no shared prefix.
PROMPT_LENS = [96, 112, 128, 144]
assert len(PROMPT_LENS) == BATCH_SIZE


def _make_prompts(rng: np.random.Generator) -> list[list[int]]:
    """Generate BATCH_SIZE prompts with distinct lengths and random content."""
    return [
        rng.integers(1000, 30000, size=length).tolist()
        for length in PROMPT_LENS
    ]


# ---------------------------------------------------------------------------
# beam_engine — batch prefill + batch decode
# ---------------------------------------------------------------------------

def _be_run_once_batch(
    model,
    config,
    prompts: list[list[int]],
    output_len: int,
) -> tuple[float, list[float]]:
    """One full batch prefill + batch decode run. Returns (prefill_s, list[decode_s])."""
    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    num_layers = config.num_hidden_layers
    batch_size = len(prompts)
    prompt_lens = [len(p) for p in prompts]

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

    # Allocate KV-cache pages for each prompt
    seq_pages: list[list[int]] = []
    for plen in prompt_lens:
        num_pages = (plen + PAGE_SIZE - 1) // PAGE_SIZE
        seq_pages.append([page_table.allocate_block() for _ in range(num_pages)])

    # --- Build batch prefill wrapper metadata ---

    # qo_indptr: cumulative query token counts [0, len0, len0+len1, ...]
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    for i, l in enumerate(prompt_lens):
        qo_indptr[i + 1] = qo_indptr[i] + l

    # paged_kv_indptr: cumulative page counts across sequences
    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    for i, pages in enumerate(seq_pages):
        paged_kv_indptr[i + 1] = paged_kv_indptr[i] + len(pages)

    paged_kv_indices = torch.tensor(
        [p for pages in seq_pages for p in pages], dtype=torch.int32, device=DEVICE
    )
    paged_kv_last_page_len = torch.tensor(
        [plen - (len(pages) - 1) * PAGE_SIZE
         for plen, pages in zip(prompt_lens, seq_pages)],
        dtype=torch.int32, device=DEVICE,
    )

    # Per-token KV write positions (packed across all sequences)
    kv_page_indices_list: list[int] = []
    kv_page_offsets_list: list[int] = []
    for plen, pages in zip(prompt_lens, seq_pages):
        for i in range(plen):
            kv_page_indices_list.append(pages[i // PAGE_SIZE])
            kv_page_offsets_list.append(i % PAGE_SIZE)
    kv_page_indices = torch.tensor(kv_page_indices_list, dtype=torch.int32, device=DEVICE)
    kv_page_offsets = torch.tensor(kv_page_offsets_list, dtype=torch.int32, device=DEVICE)

    prefill_wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=PAGE_SIZE,
        causal=True,
    )

    # Pack all prompt tokens into [1, total_tokens] for the forward pass
    all_token_ids = [t for prompt in prompts for t in prompt]
    input_ids = torch.tensor(all_token_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    # Positions: each sequence restarts from 0
    positions = torch.cat([
        torch.arange(plen, device=DEVICE) for plen in prompt_lens
    ]).unsqueeze(0)  # [1, total_tokens]

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
        # hidden_states: [1, total_tokens, hidden] — extract last token per sequence
        seq_end_indices = (qo_indptr[1:] - 1).long()          # [batch_size]
        last_hidden = hidden_states[0, seq_end_indices, :]     # [batch_size, hidden]
        logits = model.compute_logits(last_hidden)             # [batch_size, vocab]
        next_tokens = logits.argmax(dim=-1, keepdim=True)      # [batch_size, 1]
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0

    # current_positions[i]: position to write the next token for seq i
    current_positions = list(prompt_lens)

    # --- Decode loop ---
    decode_times: list[float] = []

    with torch.no_grad():
        for _ in range(output_len - 1):
            # Extend page lists where needed
            for i, pos in enumerate(current_positions):
                if pos % PAGE_SIZE == 0:
                    seq_pages[i].append(page_table.allocate_block())

            # Build decode wrapper metadata
            decode_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
            for i, pages in enumerate(seq_pages):
                decode_indptr[i + 1] = decode_indptr[i] + len(pages)

            decode_indices = torch.tensor(
                [p for pages in seq_pages for p in pages], dtype=torch.int32, device=DEVICE
            )
            decode_last_page_len = torch.tensor(
                [(pos % PAGE_SIZE) + 1 for pos in current_positions],
                dtype=torch.int32, device=DEVICE,
            )

            decode_wrapper.plan(
                indptr=decode_indptr,
                indices=decode_indices,
                last_page_len=decode_last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=PAGE_SIZE,
            )

            # Write positions for the new token in each sequence
            write_page_indices = torch.tensor(
                [seq_pages[i][pos // PAGE_SIZE] for i, pos in enumerate(current_positions)],
                dtype=torch.int32, device=DEVICE,
            )
            write_page_offsets = torch.tensor(
                [pos % PAGE_SIZE for pos in current_positions],
                dtype=torch.int32, device=DEVICE,
            )

            attn_metadata = AttentionMetadata(
                is_prefill=False,
                prefill_wrapper=None,
                decode_wrapper=decode_wrapper,
                page_table=page_table,
                kv_page_indices=write_page_indices,
                kv_page_offsets=write_page_offsets,
            )

            decode_positions = torch.tensor(
                [[pos] for pos in current_positions], device=DEVICE
            )  # [batch_size, 1]

            torch.cuda.synchronize()
            t_step = time.perf_counter()
            hidden_states = model.forward(
                input_ids=next_tokens,       # [batch_size, 1]
                positions=decode_positions,  # [batch_size, 1]
                attn_metadata=attn_metadata,
            )
            logits = model.compute_logits(hidden_states[:, -1, :])  # [batch_size, vocab]
            next_tokens = logits.argmax(dim=-1, keepdim=True)        # [batch_size, 1]
            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - t_step)

            current_positions = [pos + 1 for pos in current_positions]

    return prefill_time, decode_times


def run_beam_engine_benchmark(prompts: list[list[int]], output_len: int,
                               num_warmup: int, num_iters: int):
    print("Loading beam_engine model...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("beam_engine model loaded.\n")

    print(f"[beam_engine] Warming up ({num_warmup} iters)...")
    for _ in range(num_warmup):
        _be_run_once_batch(model, config, prompts, output_len)

    print(f"[beam_engine] Benchmarking ({num_iters} iters)...")
    all_prefill: list[float] = []
    all_decode: list[float] = []
    for i in range(num_iters):
        p, d = _be_run_once_batch(model, config, prompts, output_len)
        all_prefill.append(p)
        all_decode.extend(d)
        print(f"  iter {i + 1:2d}/{num_iters}: prefill={p * 1e3:.1f} ms  "
              f"decode_avg={np.mean(d) * 1e3:.2f} ms/step")

    del model
    torch.cuda.empty_cache()

    return np.mean(all_prefill) * 1e3, np.mean(all_decode) * 1e3  # ms


# ---------------------------------------------------------------------------
# vllm — batch
# ---------------------------------------------------------------------------

def run_vllm_benchmark(prompts: list[list[int]], output_len: int,
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
    # Each prompt has distinct token ids and a different length, so vLLM
    # cannot reuse KV cache across sequences even if prefix caching were on.
    vllm_prompts = [{"prompt_token_ids": ids} for ids in prompts]

    def run_once() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(vllm_prompts, sampling_params=sampling_params, use_tqdm=False)
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
    rng = np.random.default_rng(42)
    prompts = _make_prompts(rng)

    input_lens_str = ", ".join(str(len(p)) for p in prompts)
    total_input = sum(len(p) for p in prompts)

    print("=" * 65)
    print("  Benchmark: beam_engine vs vllm  (batch decoding)")
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Batch size: {BATCH_SIZE} sequences")
    print(f"  Input lens: [{input_lens_str}] tokens  (total {total_input})")
    print(f"  Output:     {OUTPUT_LEN} tokens/seq")
    print(f"  Warmup:     {NUM_WARMUP}  |  Bench iters: {NUM_ITERS}")
    print("=" * 65)

    be_prefill_ms, be_decode_ms = run_beam_engine_benchmark(
        prompts, OUTPUT_LEN, NUM_WARMUP, NUM_ITERS
    )

    vllm_total_ms = run_vllm_benchmark(
        prompts, OUTPUT_LEN, NUM_WARMUP, NUM_ITERS
    )

    be_total_ms = be_prefill_ms + be_decode_ms * (OUTPUT_LEN - 1)

    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  beam_engine  (batch={BATCH_SIZE})")
    print(f"    Prefill latency:        {be_prefill_ms:8.2f} ms")
    print(f"    Decode latency / step:  {be_decode_ms:8.2f} ms")
    print(f"    Total est:              {be_total_ms:8.2f} ms")
    print(f"  vllm  (FlashInfer, eager, batch={BATCH_SIZE})")
    print(f"    Total latency:          {vllm_total_ms:8.2f} ms")
    print(f"  Ratio beam_engine / vllm: {be_total_ms / vllm_total_ms:8.2f}x")
    print("=" * 65)

"""
Benchmark beam_engine vs vllm on Llama-3.1-8B — beam search.

vllm is configured with:
  - AttentionBackendEnum.FLASHINFER  (same kernel family as beam_engine)
  - enforce_eager=True               (no torch.compile / CUDA graphs)

Each sequence uses a distinct prompt (different content and different length) to
prevent vLLM prefix-caching from giving it an unfair advantage.

beam_engine runs beam search per-prompt sequentially (no cross-prompt batching).
vllm uses its built-in llm.beam_search() method.
"""

import time

import numpy as np
import torch
from transformers import AutoTokenizer

from test_beam_search import beam_search

from beam_engine.models.modeling_llama import LlamaForCausalLM
from vllm import LLM
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.sampling_params import BeamSearchParams

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16
PAGE_SIZE = 16

BATCH_SIZE = 4
OUTPUT_LEN = 128
NUM_WARMUP = 3
NUM_ITERS = 10

# Distinct prompt lengths (tokens) — one per batch slot.
PROMPT_LENS = [96, 112, 128, 144]
assert len(PROMPT_LENS) == BATCH_SIZE

BEAM_WIDTH = 4


def _make_prompts(rng: np.random.Generator) -> list[list[int]]:
    """Generate BATCH_SIZE prompts with distinct lengths and random content."""
    return [
        rng.integers(1000, 30000, size=length).tolist()
        for length in PROMPT_LENS
    ]


# ---------------------------------------------------------------------------
# beam_engine — sequential per-prompt beam search
# ---------------------------------------------------------------------------

def _be_run_once(
    model,
    config,
    prompts: list[list[int]],
    output_len: int,
    beam_width: int,
    collect_tokens: bool = False,
) -> tuple[float, list[list[int]] | None]:
    """Run beam search on each prompt sequentially.

    Returns (total_time_s, list[best_beam_tokens] | None).
    """
    total_time = 0.0
    best_tokens: list[list[int]] | None = [] if collect_tokens else None

    for prompt in prompts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        beams = beam_search(model, config, prompt, output_len, beam_width)
        torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        if collect_tokens:
            best_tokens.append(beams[0].token_ids)

    return total_time, best_tokens


def run_beam_engine_benchmark(
    prompts: list[list[int]],
    output_len: int,
    beam_width: int,
    num_warmup: int,
    num_iters: int,
    tokenizer,
) -> float:
    """Load model, verify, warmup, benchmark. Returns avg total latency in ms."""
    print("Loading beam_engine model...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("beam_engine model loaded.\n")

    print("[beam_engine] Verification pass (best beam, first 20 tokens per sequence):")
    _, gen = _be_run_once(model, config, prompts, output_len, beam_width, collect_tokens=True)
    for i, tokens in enumerate(gen):
        text = tokenizer.decode(tokens[:20], skip_special_tokens=True)
        print(f"  seq[{i}] (prompt_len={len(prompts[i])}): {text!r}")
    print()

    print(f"[beam_engine] Warming up ({num_warmup} iters)...")
    for _ in range(num_warmup):
        _be_run_once(model, config, prompts, output_len, beam_width)

    print(f"[beam_engine] Benchmarking ({num_iters} iters)...")
    latencies: list[float] = []
    for i in range(num_iters):
        t, _ = _be_run_once(model, config, prompts, output_len, beam_width)
        latencies.append(t)
        print(f"  iter {i + 1:2d}/{num_iters}: total={t * 1e3:.1f} ms")

    del model
    torch.cuda.empty_cache()

    return np.mean(latencies) * 1e3  # ms


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
) -> float:
    """Load LLM, verify, warmup, benchmark. Returns avg total latency in ms."""
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
    for i, out in enumerate(outputs):
        # output tokens include prompt; strip prompt to get generated tokens
        gen_tokens = out.sequences[0].tokens[len(prompts[i]):]
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

    return np.mean(latencies) * 1e3  # ms


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

    print("=" * 65)
    print(f"  Benchmark: beam_engine vs vllm  (beam search, width={BEAM_WIDTH})")
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Batch size: {BATCH_SIZE} sequences")
    print(f"  Beam width: {BEAM_WIDTH}")
    print(f"  Input lens: [{input_lens_str}] tokens  (total {total_input})")
    print(f"  Output:     {OUTPUT_LEN} tokens/seq")
    print(f"  Warmup:     {NUM_WARMUP}  |  Bench iters: {NUM_ITERS}")
    print("=" * 65)

    be_total_ms = run_beam_engine_benchmark(
        prompts, OUTPUT_LEN, BEAM_WIDTH, NUM_WARMUP, NUM_ITERS, tokenizer
    )

    vllm_total_ms = run_vllm_benchmark(
        prompts, OUTPUT_LEN, BEAM_WIDTH, NUM_WARMUP, NUM_ITERS, tokenizer
    )

    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  beam_engine  (beam_width={BEAM_WIDTH}, batch={BATCH_SIZE}, sequential)")
    print(f"    Total latency:          {be_total_ms:8.2f} ms")
    print(f"  vllm  (FlashInfer, eager, beam_width={BEAM_WIDTH}, batch={BATCH_SIZE})")
    print(f"    Total latency:          {vllm_total_ms:8.2f} ms")
    print(f"  Ratio beam_engine / vllm: {be_total_ms / vllm_total_ms:.2f}x")
    print("=" * 65)

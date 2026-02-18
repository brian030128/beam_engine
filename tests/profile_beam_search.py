"""
Torch profiler for beam_engine vs vLLM beam search.

Produces Chrome trace files for visual inspection in Perfetto UI
(chrome://tracing or https://ui.perfetto.dev/).

Uses a single prompt (len=96) to keep trace files manageable.
"""

import os

# Must be set before importing vLLM so it picks up the profiler config.
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from test_beam_search import beam_search

from beam_engine.models.modeling_llama import LlamaForCausalLM
from vllm import LLM
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.sampling_params import BeamSearchParams

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16
PAGE_SIZE = 16
BEAM_WIDTH = 4
OUTPUT_LEN = 128
PROMPT_LENS = [96, 112, 128, 144]


def _make_prompts(rng: np.random.Generator) -> list[list[int]]:
    """Generate prompts with distinct lengths and random content."""
    return [
        rng.integers(1000, 30000, size=length).tolist()
        for length in PROMPT_LENS
    ]


def profile_beam_engine(prompt: list[int], output_len: int, beam_width: int):
    print("Loading beam_engine model...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("beam_engine model loaded.\n")

    # Warmup
    print("[beam_engine] Warmup...")
    beam_search(model, config, prompt, output_len, beam_width)
    torch.cuda.synchronize()

    # Profile
    print("[beam_engine] Profiling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        beam_search(model, config, prompt, output_len, beam_width)
        torch.cuda.synchronize()

    prof.export_chrome_trace("beam_engine_beam_search.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    del model
    torch.cuda.empty_cache()


def profile_vllm(prompt: list[int], output_len: int, beam_width: int):
    print("\nLoading vllm model (FlashInfer backend, eager mode)...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        enforce_eager=True,
        attention_backend=AttentionBackendEnum.FLASHINFER,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
    )
    print("vllm model loaded.\n")

    params = BeamSearchParams(
        beam_width=beam_width,
        max_tokens=output_len,
        ignore_eos=True,
        temperature=0.0,
    )

    # Warmup
    print("[vllm] Warmup...")
    llm.beam_search([{"prompt_token_ids": prompt}], params)
    torch.cuda.synchronize()

    # Profile using vLLM's built-in torch profiler integration.
    # Traces are written to VLLM_TORCH_PROFILER_DIR (set at top of file).
    print("[vllm] Profiling...")
    llm.start_profile()
    llm.beam_search([{"prompt_token_ids": prompt}], params)
    torch.cuda.synchronize()
    llm.stop_profile()

    print(f"[vllm] Traces written to {os.environ['VLLM_TORCH_PROFILER_DIR']}/")

    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    prompts = _make_prompts(rng)
    prompt = prompts[0]  # single prompt, len=96

    print("=" * 65)
    print(f"  Profiling: beam_engine vs vLLM  (beam search, width={BEAM_WIDTH})")
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Prompt len: {len(prompt)} tokens")
    print(f"  Output:     {OUTPUT_LEN} tokens")
    print(f"  Beam width: {BEAM_WIDTH}")
    print("=" * 65)

    profile_beam_engine(prompt, OUTPUT_LEN, BEAM_WIDTH)
    profile_vllm(prompt, OUTPUT_LEN, BEAM_WIDTH)

    print("\nDone. Trace files written:")
    print("  beam_engine: beam_engine_beam_search.json")
    print("  vllm:        ./vllm_profile/")

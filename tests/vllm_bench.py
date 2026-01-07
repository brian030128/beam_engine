# run_vllm.py
"""
vLLM Benchmark with Profiling Support

To enable profiling, set the VLLM_TORCH_PROFILER_DIR environment variable:

    Windows:
    set VLLM_TORCH_PROFILER_DIR=./vllm_traces
    python tests/vllm_bench.py

    Linux/Mac:
    VLLM_TORCH_PROFILER_DIR=./vllm_traces python tests/vllm_bench.py

The traces can be visualized at: https://ui.perfetto.dev/
"""
import os
import time
from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import BeamSearchParams

MODEL_NAME = "meta-llama/Llama-3.1-8B"
PROMPT = "The future of artificial intelligence is" * 50

BEAM_SIZE = 8
MAX_NEW_TOKENS = 20
NUM_RETURN_SEQS = 4
TEMPERATURE = 1.0


def run_vllm_beam_search():
    # Check if profiling is enabled
    profiler_dir = os.environ.get('VLLM_TORCH_PROFILER_DIR')
    if profiler_dir:
        print(f"[vLLM] Profiling ENABLED - traces will be saved to: {profiler_dir}")
        print(f"[vLLM] Visualize at: https://ui.perfetto.dev/")
    else:
        print("[vLLM] Profiling DISABLED - set VLLM_TORCH_PROFILER_DIR to enable")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    prompt_token_ids = tokenizer.encode(PROMPT)
    prompt_len = len(prompt_token_ids)

    print(f"[vLLM] Prompt length: {prompt_len}")
    print(f"[vLLM] Max new tokens: {MAX_NEW_TOKENS}")

    # vLLM engine starts here
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
    )

    params = BeamSearchParams(
        beam_width=BEAM_SIZE,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        ignore_eos=True,
    )

    # Warmup
    print("[vLLM] Warming up (3 iterations)...")
    for _ in range(3):
        _ = llm.beam_search(
            prompts=[{"prompt": PROMPT}],
            params=params,
        )
    print("[vLLM] Warmup complete. Starting benchmark...")

    start = time.perf_counter()

    outputs = llm.beam_search(
        prompts=[{"prompt": PROMPT}],
        params=params,
    )

    elapsed = time.perf_counter() - start

    print(f"\n[vLLM] Generation time: {elapsed:.4f}s")
    print(f"[vLLM] Number of outputs: {len(outputs)}")

    for out in outputs:
        for i, seq in enumerate(out.sequences):
            print(f"\n[vLLM seq {i+1}]")
            print(seq.text[:300], "...")

    if profiler_dir:
        print("\n" + "=" * 80)
        print("PROFILING COMPLETE")
        print("=" * 80)
        print(f"Trace files saved to: {profiler_dir}")
        print(f"Open https://ui.perfetto.dev/ and load the trace files")
        print("=" * 80)

    print("\n[vLLM] DONE — exiting process")


if __name__ == "__main__":
    run_vllm_beam_search()
    # 💀 Process exit = GPU fully released


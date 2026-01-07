# run_vllm.py
"""
vLLM Benchmark with Profiling Support

Profiling is enabled by default and saves traces to ./vllm_profile/
The traces can be visualized at: https://ui.perfetto.dev/
"""
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

# Profiling configuration
ENABLE_PROFILING = True
PROFILER_OUTPUT_DIR = "./vllm_profile"
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

def run_vllm_beam_search():
    if ENABLE_PROFILING:
        print(f"[vLLM] Profiling ENABLED - traces will be saved to: {PROFILER_OUTPUT_DIR}")
        print(f"[vLLM] Visualize at: https://ui.perfetto.dev/")
    else:
        print("[vLLM] Profiling DISABLED")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    prompt_token_ids = tokenizer.encode(PROMPT)
    prompt_len = len(prompt_token_ids)

    print(f"[vLLM] Prompt length: {prompt_len}")
    print(f"[vLLM] Max new tokens: {MAX_NEW_TOKENS}")

    # vLLM engine with profiling config
    profiler_config = None
    if ENABLE_PROFILING:
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": PROFILER_OUTPUT_DIR,
        }

    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        #profiler_config=profiler_config, not that version yet
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
    print("[vLLM] Warmup complete. Starting profiling...")

    # Start profiling
    if ENABLE_PROFILING:
        llm.start_profile()

    start = time.perf_counter()

    outputs = llm.beam_search(
        prompts=[{"prompt": PROMPT}],
        params=params,
    )

    elapsed = time.perf_counter() - start

    # Stop profiling
    if ENABLE_PROFILING:
        llm.stop_profile()

    print(f"\n[vLLM] Generation time: {elapsed:.4f}s")
    print(f"[vLLM] Number of outputs: {len(outputs)}")

    for out in outputs:
        for i, seq in enumerate(out.sequences):
            print(f"\n[vLLM seq {i+1}]")
            print(seq.text[:300], "...")

    if ENABLE_PROFILING:
        print("\n" + "=" * 80)
        print("PROFILING COMPLETE")
        print("=" * 80)
        print(f"Trace files saved to: {PROFILER_OUTPUT_DIR}")
        print(f"Open https://ui.perfetto.dev/ and load the trace files")
        print("=" * 80)

        # Buffer time for profiler to finish writing
        print("\n[vLLM] Waiting 10 seconds for profiler to finish writing...")
        time.sleep(10)

    print("\n[vLLM] DONE — exiting process")


if __name__ == "__main__":
    run_vllm_beam_search()
    # 💀 Process exit = GPU fully released


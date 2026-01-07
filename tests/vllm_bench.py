# run_vllm.py
import time
import torch
from torch.profiler import profile, ProfilerActivity, record_function
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    prompt_token_ids = tokenizer.encode(PROMPT)
    prompt_len = len(prompt_token_ids)

    print(f"[vLLM] Prompt length: {prompt_len}")
    print(f"[vLLM] Max new tokens: {MAX_NEW_TOKENS}")

    # 🚨 vLLM engine starts here
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
    print("[vLLM] Warmup complete. Starting profiling...")

    # Profile with PyTorch profiler
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    start = time.perf_counter()

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("vllm_beam_search_generation"):
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

    # Print profiling results
    print("\n" + "=" * 80)
    print("PROFILING RESULTS - TOP 40 OPERATIONS BY CUDA TIME")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=40,
        max_name_column_width=60
    ))

    print("\n" + "=" * 80)
    print("PROFILING RESULTS - TOP 20 OPERATIONS BY CPU TIME")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=20,
        max_name_column_width=60
    ))

    print("\n" + "=" * 80)
    print("MEMORY USAGE - TOP 20 OPERATIONS")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage",
        row_limit=20,
        max_name_column_width=60
    ))

    # Save trace for chrome://tracing
    trace_file = "vllm_beam_trace.json"
    prof.export_chrome_trace(trace_file)

    print("\n" + "=" * 80)
    print("TRACE FILES SAVED")
    print("=" * 80)
    print(f"Chrome trace: {trace_file}")
    print(f"  -> Open chrome://tracing in Chrome")
    print(f"  -> Click 'Load' and select {trace_file}")
    print("=" * 80)

    print("\n[vLLM] DONE — exiting process")


if __name__ == "__main__":
    run_vllm_beam_search()
    # 💀 Process exit = GPU fully released


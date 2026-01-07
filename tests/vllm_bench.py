# run_vllm.py
import time
import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import BeamSearchParams

MODEL_NAME = "meta-llama/Llama-3.1-8B"
PROMPT = "The future of artificial intelligence is" * 100

BEAM_SIZE = 8
MAX_LENGTH = 1000
NUM_RETURN_SEQS = 4
TEMPERATURE = 1.0


def run_vllm_beam_search():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    prompt_token_ids = tokenizer.encode(PROMPT)
    prompt_len = len(prompt_token_ids)
    max_new_tokens = max(0, MAX_LENGTH - prompt_len)

    print(f"[vLLM] Prompt length: {prompt_len}")
    print(f"[vLLM] Max new tokens: {max_new_tokens}")

    # 🚨 vLLM engine starts here
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
    )

    params = BeamSearchParams(
        beam_width=BEAM_SIZE,
        max_tokens=max_new_tokens,
        temperature=TEMPERATURE,
        ignore_eos=False,
    )


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

    print("\n[vLLM] DONE — exiting process")


if __name__ == "__main__":
    run_vllm_beam_search()
    # 💀 Process exit = GPU fully released


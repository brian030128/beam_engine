"""
Benchmark beam search across different page sizes.

Measures GPU time for beam search (beam_width=4, max_new_tokens=20)
with page sizes [1, 2, 4, 8, 16, 32, 64] and prints a summary table.
"""

import torch
from transformers import AutoTokenizer

from beam_engine.models.modeling_llama import LlamaForCausalLM
from tests.test_beam_search import beam_search

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda"
DTYPE = torch.float16

PAGE_SIZES = [1, 2, 4, 8, 16, 32, 64]
BEAM_WIDTH = 4
MAX_NEW_TOKENS = 20
WARMUP_RUNS = 2
TIMED_RUNS = 5

PROMPT = "The capital of France is"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    config = model.config
    print("Model loaded!\n")

    prompt_ids = tokenizer.encode(PROMPT)

    results = []

    for ps in PAGE_SIZES:
        print(f"--- page_size={ps} ---")

        # Warmup
        for _ in range(WARMUP_RUNS):
            beam_search(
                model, config, [prompt_ids], MAX_NEW_TOKENS, BEAM_WIDTH,
                page_size=ps,
            )
        torch.cuda.synchronize()

        # Timed runs
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(TIMED_RUNS):
            beams = beam_search(
                model, config, [prompt_ids], MAX_NEW_TOKENS, BEAM_WIDTH,
                page_size=ps,
            )
        end_event.record()
        torch.cuda.synchronize()

        total_ms = start_event.elapsed_time(end_event)
        avg_ms = total_ms / TIMED_RUNS

        best_beam = beams[0][0]
        text = tokenizer.decode(best_beam.token_ids, skip_special_tokens=True)
        results.append((ps, avg_ms, text))
        print(f"  avg={avg_ms:.1f}ms  text=\"{PROMPT}{text}\"")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'page_size':>10}  {'avg_ms':>10}  {'output'}")
    print("-" * 70)
    for ps, avg_ms, text in results:
        print(f"{ps:>10}  {avg_ms:>10.1f}  {PROMPT}{text}")
    print("=" * 70)


if __name__ == "__main__":
    main()

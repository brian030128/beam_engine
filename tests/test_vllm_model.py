"""
Test script for token-by-token generation with greedy decoding using LlamaForCausalLM.
Uses FlashInfer attention kernels.
"""

import torch
from transformers import AutoTokenizer

from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading LlamaForCausalLM model...")
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
    print("Model loaded!")

    # Input prompt
    prompt = "The capital of France is"
    print(f"\nPrompt: {prompt}")

    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    # Token-by-token generation loop
    max_new_tokens = 20
    generated_tokens = []

    print("\nGenerating tokens one at a time (greedy decoding):")
    print("-" * 50)

    with torch.no_grad():
        for i in range(max_new_tokens):
            # Create position ids
            seq_len = input_ids.shape[1]
            positions = torch.arange(seq_len, device=DEVICE).unsqueeze(0)

            # Forward pass through model (no page_table for simple test)
            hidden_states = model.forward(
                input_ids=input_ids,
                positions=positions,
            )

            # Compute logits from last hidden state
            logits = model.compute_logits(hidden_states[:, -1:, :])

            # Greedy decoding: take argmax
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = next_token_id.item()
            generated_tokens.append(token_id)

            # Decode token for display
            token_text = tokenizer.decode([token_id])
            print(f"Step {i+1}: token_id={token_id}, text='{token_text}'")

            # Check for EOS
            if token_id == tokenizer.eos_token_id:
                print("  [EOS reached]")
                break

            # Append new token to input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    print("-" * 50)

    # Decode full generated sequence
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"\nFinal generated text:\n{full_text}")
    print(f"\nGenerated token ids: {generated_tokens}")


if __name__ == "__main__":
    main()

"""
Test script for token-by-token generation with greedy decoding using LlamaForCausalLM.
"""

import torch
from transformers import AutoTokenizer, LlamaConfig
from vllm.config import VllmConfig, ModelConfig, CacheConfig, SchedulerConfig, LoadConfig, ParallelConfig, DeviceConfig

from beam_engine.models.modeling_llama import LlamaForCausalLM


MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
DTYPE = torch.float16


def create_vllm_config(model_name: str) -> VllmConfig:
    """Create VllmConfig for model initialization."""
    model_config = ModelConfig(
        model=model_name,
        task="generate",
        tokenizer=model_name,
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype=DTYPE,
        seed=42,
    )

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=4,
        cache_dtype="auto",
    )

    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    scheduler_config = SchedulerConfig(
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=False,
    )

    load_config = LoadConfig()
    device_config = DeviceConfig(device=DEVICE)

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        load_config=load_config,
        device_config=device_config,
    )


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Creating vLLM config...")
    vllm_config = create_vllm_config(MODEL_NAME)

    print("Loading LlamaForCausalLM model...")
    model = LlamaForCausalLM(vllm_config=vllm_config)
    model = model.to(DEVICE).to(DTYPE)
    model.eval()

    # Load weights from HuggingFace
    print("Loading weights...")
    from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
    # For testing, we'll use the actual weights loader
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download, list_repo_files

    # Get weight files
    files = list_repo_files(MODEL_NAME)
    safetensor_files = [f for f in files if f.endswith('.safetensors')]

    weights = {}
    for sf in safetensor_files:
        path = hf_hub_download(MODEL_NAME, sf)
        weights.update(load_file(path))

    model.load_weights(weights.items())
    print("Model loaded!")

    # Input prompt
    prompt = "The capital of France is"
    print(f"\nPrompt: {prompt}")

    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    seq_len = input_ids.shape[1]

    # Token-by-token generation loop
    max_new_tokens = 20
    generated_tokens = []

    print("\nGenerating tokens one at a time (greedy decoding):")
    print("-" * 50)

    with torch.no_grad():
        for i in range(max_new_tokens):
            # Create position ids
            positions = torch.arange(input_ids.shape[1], device=DEVICE)

            # Forward pass through model
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

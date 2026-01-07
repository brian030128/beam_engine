"""
Benchmark script for comparing beam search implementations.
FIXED VERSION: Adapted for legacy vLLM (BeamSearchParams) + Fixes token limit mismatch.
"""

import torch
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM
# Using BeamSearchParams as supported by your vLLM version
from vllm.sampling_params import BeamSearchParams

from beam_engine.beam import BeamSearchGenerator
from beam_engine.beam_strategy import DiverseBeamSearchStrategy, VanillaBeamSearchStrategy
from beam_engine.models.modeling_llama import LlamaForCausalLM
from beam_engine.logger import init_logger, set_logging_level, LogLevel

logger = init_logger(__name__)

def run_huggingface_beam_search(hf_model, tokenizer, prompt: str, beam_size: int = 4,
                                max_length: int = 50, num_return_sequences: int = 1,
                                temperature: float = 1.0):
    """Run HuggingFace's native beam search for comparison."""
    logger.info("\n" + "=" * 80)
    logger.info("=== HUGGINGFACE BEAM SEARCH (REFERENCE) ===")
    logger.info("=" * 80)

    inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
    input_len = inputs.input_ids.shape[1]
    
    # Calculate how many tokens we actually expect to generate
    expected_new_tokens = max_length - input_len
    logger.info(f"Input Length: {input_len} | Max Total Length: {max_length} | Allowed New Tokens: {expected_new_tokens}")

    if expected_new_tokens <= 0:
        logger.warning("Prompt is longer than max_length! HF will generate 0 tokens.")

    # Warm up GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_length=max_length,
            num_beams=beam_size,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            do_sample=False,  # Deterministic beam search
            early_stopping=False,
            return_dict_in_generate=True,
            output_scores=True,
            repetition_penalty=1,
            length_penalty=1,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    generated_texts = []
    logger.info(f"\n[HF RESULTS] Generated {len(outputs.sequences)} sequences:")
    for idx, sequence in enumerate(outputs.sequences):
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        generated_texts.append(text)
        gen_token_count = len(sequence) - input_len
        logger.info(f"\n[HF {idx+1}] Generated Tokens: {gen_token_count}")

    logger.info(f"\n[HF TIMING] Total generation time: {elapsed_time:.4f}s")
    return generated_texts, elapsed_time




def demo_diverse_beam_search(model, tokenizer, model_name, device):
    """Demonstrate diverse beam search generation with sequential model loading."""
    logger.info("=== Diverse Beam Search Demo ===")

    # Example prompts
    prompts = [
        "The future of artificial intelligence is" * 100, 
    ]

    for prompt in prompts:
        logger.info(f"\nPrompt Length (approx chars): {len(prompt)}")
        logger.info("-" * 50)

        
        # ---------------------------------------------------------
        # 1. HuggingFace Benchmark
        # ---------------------------------------------------------
        logger.info("\n[BENCHMARK] Loading HuggingFace model...")
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

        hf_texts, hf_time = run_huggingface_beam_search(
            hf_model=hf_model,
            tokenizer=tokenizer,
            prompt=prompt,
            beam_size=8,
            max_length=1000,
            num_return_sequences=4,
            temperature=1.0
        )

        # Cleanup HuggingFace
        logger.info("[BENCHMARK] Cleaning up HuggingFace model...")
        del hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------------------------------------------------------
        # 3. Custom Benchmark
        # ---------------------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("=== CUSTOM BEAM SEARCH (CASCADE ATTENTION) ===")
        logger.info("=" * 80)

        logger.info(f"Loading custom model from {model_name}...")
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        
        generator = BeamSearchGenerator(model, tokenizer, DiverseBeamSearchStrategy())
        generated_texts = generator.generate(
            input_text=prompt,
            beam_size=8,
            max_length=1000,
            num_return_sequences=4,
            temperature=1.0
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        generator = BeamSearchGenerator(model, tokenizer, VanillaBeamSearchStrategy())
        generated_texts = generator.generate(
            input_text=prompt,
            beam_size=8,
            max_length=1000, 
            num_return_sequences=4,
            temperature=1.0
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        custom_time = end_time - start_time

        logger.info(f"\n[CUSTOM TIMING] Total generation time: {custom_time:.4f}s")

        # ---------------------------------------------------------
        # 4. Results
        # ---------------------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("=== PERFORMANCE BENCHMARK ===")
        logger.info("=" * 80)
        logger.info(f"HuggingFace time:  {hf_time:.4f}s")
        logger.info(f"vLLM time:         {vllm_time:.4f}s")
        logger.info(f"Custom time:       {custom_time:.4f}s")

        hf_vs_custom = hf_time / custom_time
        vllm_vs_custom = vllm_time / custom_time
        vllm_vs_hf = hf_time / vllm_time

        logger.info(f"\nCustom vs HuggingFace: {hf_vs_custom:.2f}x {'faster' if hf_vs_custom > 1 else 'slower'}")
        logger.info(f"Custom vs vLLM:        {vllm_vs_custom:.2f}x {'faster' if vllm_vs_custom > 1 else 'slower'}")
        logger.info(f"vLLM vs HuggingFace:   {vllm_vs_hf:.2f}x {'faster' if vllm_vs_hf > 1 else 'slower'}")
        logger.info("=" * 80)


if __name__ == "__main__":
    # Set logging level to INFO (hides DEBUG messages)
    set_logging_level(LogLevel.INFO)

    logger.info("Loading custom model and tokenizer...")

    # Model and tokenizer setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    demo_diverse_beam_search(model, tokenizer, model_name, device)

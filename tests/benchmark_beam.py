"""
Benchmark script for comparing beam search implementations.

This module provides benchmarking utilities to compare:
- HuggingFace native beam search
- vLLM beam search
- Custom cascade attention beam search
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM
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
            early_stopping=True,
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
        tokens = sequence.tolist()
        logger.info(f"\n[HF {idx+1}] Tokens: {len(tokens)}")
        logger.info(f"  Full token sequence: {tokens}")
        logger.info(f"  Text: {text}")

    logger.info(f"\n[HF TIMING] Total generation time: {elapsed_time:.4f}s ({elapsed_time*1000:.2f}ms)")

    return generated_texts, elapsed_time


def run_vllm_beam_search(vllm_model, tokenizer, prompt: str, beam_size: int = 4,
                         max_length: int = 50, num_return_sequences: int = 1,
                         temperature: float = 1.0):
    """Run vLLM's beam search for comparison."""
    logger.info("\n" + "=" * 80)
    logger.info("=== VLLM BEAM SEARCH (REFERENCE) ===")
    logger.info("=" * 80)

    # Warm up GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    # Create BeamSearchParams
    beam_params = BeamSearchParams(
        beam_width=beam_size,
        max_tokens=max_length,
        temperature=temperature,
    )

    # Run beam search
    outputs = vllm_model.beam_search(
        prompts=[{"prompt": prompt}],
        params=beam_params
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    generated_texts = []
    logger.info(f"\n[VLLM RESULTS] Generated {len(outputs)} outputs:")
    for output_idx, output in enumerate(outputs):
        logger.info(f"\n[VLLM Output {output_idx+1}] {len(output.sequences)} sequences:")
        for seq_idx, sequence in enumerate(output.sequences):
            text = sequence.text
            generated_texts.append(text)
            logger.info(f"  [Sequence {seq_idx+1}] Text: {text}")

    logger.info(f"\n[VLLM TIMING] Total generation time: {elapsed_time:.4f}s ({elapsed_time*1000:.2f}ms)")

    return generated_texts, elapsed_time


def demo_diverse_beam_search(model, tokenizer, model_name, device):
    """Demonstrate diverse beam search generation with sequential model loading."""
    logger.info("=== Diverse Beam Search Demo ===")

    # Example prompts
    prompts = [
        "The future of artificial intelligence is" * 100,
        #"Once upon a time in a magical forest,",
        #"The best way to solve climate change is"
    ]

    for prompt in prompts:
        logger.info(f"\nPrompt: '{prompt}'")
        logger.info("-" * 50)

        # Run HuggingFace beam search first
        logger.info("\n[BENCHMARK] Loading HuggingFace model...")
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

        hf_texts, hf_time = run_huggingface_beam_search(
            hf_model=hf_model,
            tokenizer=tokenizer,
            prompt=prompt,
            beam_size=8,
            max_length=1000,
            num_return_sequences=4,
            temperature=1.0
        )

        # Cleanup HuggingFace model
        logger.info("[BENCHMARK] Cleaning up HuggingFace model...")
        del hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run vLLM beam search
        logger.info("\n[BENCHMARK] Loading vLLM model...")
        vllm_model = LLM(model=model_name, dtype="float16")

        vllm_texts, vllm_time = run_vllm_beam_search(
            vllm_model=vllm_model,
            tokenizer=tokenizer,
            prompt=prompt,
            beam_size=8,
            max_length=1000,
            num_return_sequences=4,
            temperature=1.0
        )

        # Cleanup vLLM model
        logger.info("[BENCHMARK] Cleaning up vLLM model...")
        del vllm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Generate diverse sequences with custom implementation
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
            max_length=500,
            num_return_sequences=4,
            temperature=1.0
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        custom_time = end_time - start_time

        logger.info(f"\n[CUSTOM TIMING] Total generation time: {custom_time:.4f}s ({custom_time*1000:.2f}ms)")

        logger.info("\n" + "=" * 80)
        logger.info("=== COMPARISON ===")
        logger.info("=" * 80)

        logger.info("\nHuggingFace Results:")
        for i, text in enumerate(hf_texts, 1):
            logger.info(f"  {i}. {text}")

        logger.info("\nvLLM Results:")
        for i, text in enumerate(vllm_texts, 1):
            logger.info(f"  {i}. {text}")

        logger.info("\nCustom Cascade Results:")
        for i, text in enumerate(generated_texts, 1):
            logger.info(f"  {i}. {text}")

        # Performance comparison
        logger.info("\n" + "=" * 80)
        logger.info("=== PERFORMANCE BENCHMARK ===")
        logger.info("=" * 80)
        logger.info(f"HuggingFace time:  {hf_time:.4f}s ({hf_time*1000:.2f}ms)")
        logger.info(f"vLLM time:         {vllm_time:.4f}s ({vllm_time*1000:.2f}ms)")
        logger.info(f"Custom time:       {custom_time:.4f}s ({custom_time*1000:.2f}ms)")

        # Speedup comparisons
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
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = None

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Custom model loaded successfully!\n")
    logger.info("Note: HuggingFace and vLLM models will be loaded sequentially during benchmarking\n")

    # Run demonstrations with comparison
    demo_diverse_beam_search(model, tokenizer, model_name, device)

    logger.info("\n=== Strategy Comparison ===")
    logger.info("Vanilla beam search: Selects candidates purely by score - simple and fast.")
    logger.info("Diverse beam search: Promotes variety by grouping beams and penalizing similarity.")
    logger.info("\nBoth strategies are now cleanly separated with no shared dependencies!")
    logger.info("You can easily switch strategies by changing the BeamStrategy class.")
    logger.info("Try implementing TopKBeamStrategy, NucleusBeamStrategy, or other custom approaches.")

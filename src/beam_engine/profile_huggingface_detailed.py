#!/usr/bin/env python3
"""
Profile beam.py with PyTorch profiler to see GPU kernel details.
Run this instead of beam.py to get detailed profiling.
"""

import torch
from torch.profiler import profile, ProfilerActivity, record_function
import sys
import os

# Import your beam code
from beam import BeamSearchGenerator, demo_diverse_beam_search, run_huggingface_beam_search
from beam_strategy import VanillaBeamSearchStrategy
from models.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from logger import set_logging_level, LogLevel, init_logger

logger = init_logger(__name__)

def main():
    """Run beam search with detailed profiling."""

    # Set logging
    set_logging_level(LogLevel.INFO)

    logger.info("="*100)
    logger.info("PROFILING MODE - Detailed GPU kernel analysis")
    logger.info("="*100)

    # Setup (same as beam.py)
    device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
    model_name = "meta-llama/Llama-3.1-8B"

    logger.info(f"Loading custom model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device)

    logger.info(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Models loaded!\n")

    # Create generator
    strategy = VanillaBeamSearchStrategy()
    generator = BeamSearchGenerator(model, tokenizer, strategy)

    prompt = "The future of artificial intelligence is"

    logger.info(f"Prompt: '{prompt}'")
    logger.info("="*100)

    # Warmup (important!)
    logger.info("Warming up (3 iterations)...")
    for _ in range(3):
        with torch.no_grad():
            hf_texts, hf_time = run_huggingface_beam_search(
                hf_model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                beam_size=8,
                max_length=50,
                num_return_sequences=4,
                temperature=1.0
            )

    logger.info("Warmup complete. Starting profiling...")
    logger.info("="*100 + "\n")

    # Profile with PyTorch profiler
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("run_huggingface_beam_search"):
            hf_texts, hf_time = run_huggingface_beam_search(
                hf_model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                beam_size=8,
                max_length=50,
                num_return_sequences=4,
                temperature=1.0
            )

    # Print results
    logger.info("\n" + "="*100)
    logger.info("PROFILING RESULTS - TOP 40 OPERATIONS BY CUDA TIME")
    logger.info("="*100)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=40,
        max_name_column_width=60
    ))

    logger.info("\n" + "="*100)
    logger.info("PROFILING RESULTS - TOP 20 OPERATIONS BY CPU TIME")
    logger.info("="*100)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=20,
        max_name_column_width=60
    ))

    logger.info("\n" + "="*100)
    logger.info("MEMORY USAGE - TOP 20 OPERATIONS")
    logger.info("="*100)
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage",
        row_limit=20,
        max_name_column_width=60
    ))

    # Save trace for chrome://tracing
    trace_file = "hf_decode_trace.json"
    prof.export_chrome_trace(trace_file)

    logger.info("\n" + "="*100)
    logger.info("TRACE FILES SAVED")
    logger.info("="*100)
    logger.info(f"Chrome trace: {trace_file}")
    logger.info(f"  -> Open chrome://tracing in Chrome")
    logger.info(f"  -> Click 'Load' and select {trace_file}")
    logger.info(f"  -> See beautiful GPU timeline!")
    logger.info("="*100)

    # Analysis hints
    logger.info("\n" + "="*100)
    logger.info("WHAT TO LOOK FOR IN THE RESULTS ABOVE:")
    logger.info("="*100)
    logger.info("1. Large CUDA kernels (>10ms) - these are your bottlenecks")
    logger.info("2. 'torch.tensor' or 'to()' with high CPU time - data conversion overhead")
    logger.info("3. 'cudaMemcpy' operations - device transfers")
    logger.info("4. FlashInfer kernels - attention computation")
    logger.info("5. 'aten::*' operations - PyTorch ops (matmul, etc.)")
    logger.info("="*100)

    return generated_texts


if __name__ == "__main__":
    main()

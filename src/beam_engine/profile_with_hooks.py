#!/usr/bin/env python3
"""
Non-invasive profiling using PyTorch hooks.
Automatically wraps model modules with profiling markers without modifying source code.
"""

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from contextlib import contextmanager

from beam import BeamSearchGenerator
from beam_strategy import VanillaBeamSearchStrategy
from models.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer
from logger import set_logging_level, LogLevel, init_logger

logger = init_logger(__name__)


class ProfilingHooks:
    """Manages profiling hooks for non-invasive model profiling."""

    def __init__(self):
        self.handles = []

    def register_profiling_hooks(self, model):
        """Register forward hooks on all modules to add profiling markers."""

        def make_forward_pre_hook(name):
            def hook(module, input):
                # Store the record_function context on the module
                module._profiling_context = record_function(name)
                module._profiling_context.__enter__()
            return hook

        def make_forward_hook(name):
            def hook(module, input, output):
                # Exit the record_function context
                if hasattr(module, '_profiling_context'):
                    module._profiling_context.__exit__(None, None, None)
                    del module._profiling_context
            return hook

        # Register hooks for each module type we care about
        for name, module in model.named_modules():
            # Clean up the name for readability
            clean_name = name if name else "model"

            # Register pre-hook to start profiling
            pre_handle = module.register_forward_pre_hook(
                make_forward_pre_hook(clean_name)
            )
            # Register post-hook to end profiling
            post_handle = module.register_forward_hook(
                make_forward_hook(clean_name)
            )

            self.handles.append(pre_handle)
            self.handles.append(post_handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def main():
    """Run beam search with non-invasive profiling."""

    set_logging_level(LogLevel.INFO)

    logger.info("="*100)
    logger.info("NON-INVASIVE PROFILING MODE - Using module hooks")
    logger.info("="*100)

    # Setup
    device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
    model_name = "meta-llama/Llama-3.2-3B"

    logger.info(f"Loading model from {model_name}...")
    model = LlamaForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device)

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

    # Warmup
    logger.info("Warming up (3 iterations)...")
    for _ in range(3):
        with torch.no_grad():
            _ = generator.generate(
                input_text=prompt,
                beam_size=8,
                max_length=50,
                num_return_sequences=1,
                temperature=1.0
            )

    logger.info("Warmup complete. Starting profiling...")
    logger.info("="*100 + "\n")

    # Register profiling hooks
    profiling_hooks = ProfilingHooks()
    profiling_hooks.register_profiling_hooks(model)

    # Profile with PyTorch profiler
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,  # Disable to reduce overhead
        with_stack=False,       # Disable to reduce overhead
    ) as prof:
        with record_function("beam_search_generation"):
            generated_texts = generator.generate(
                input_text=prompt,
                beam_size=8,
                max_length=50,
                num_return_sequences=4,
                temperature=1.0
            )

    # Remove hooks
    profiling_hooks.remove_hooks()

    # Print results grouped by module type
    logger.info("\n" + "="*100)
    logger.info("PROFILING RESULTS - BY MODULE TYPE")
    logger.info("="*100)

    # Get events and group by module name pattern
    events = prof.key_averages()

    # Group events by pattern
    attention_time = 0
    mlp_time = 0
    norm_time = 0
    embedding_time = 0
    lm_head_time = 0
    other_time = 0

    for event in events:
        name = event.key
        cuda_time = event.cuda_time_total / 1000  # Convert to ms

        if 'self_attn' in name or 'cascade' in name or 'attention' in name:
            attention_time += cuda_time
        elif 'mlp' in name:
            mlp_time += cuda_time
        elif 'norm' in name:
            norm_time += cuda_time
        elif 'embed' in name:
            embedding_time += cuda_time
        elif 'lm_head' in name:
            lm_head_time += cuda_time
        else:
            other_time += cuda_time

    total = attention_time + mlp_time + norm_time + embedding_time + lm_head_time + other_time

    print(f"\n{'Component':<30} {'Time (ms)':<15} {'Percentage':<15}")
    print("=" * 60)
    print(f"{'Attention':<30} {attention_time:<15.2f} {attention_time/total*100:<15.1f}%")
    print(f"{'MLP':<30} {mlp_time:<15.2f} {mlp_time/total*100:<15.1f}%")
    print(f"{'Layer Norms':<30} {norm_time:<15.2f} {norm_time/total*100:<15.1f}%")
    print(f"{'Embeddings':<30} {embedding_time:<15.2f} {embedding_time/total*100:<15.1f}%")
    print(f"{'LM Head':<30} {lm_head_time:<15.2f} {lm_head_time/total*100:<15.1f}%")
    print(f"{'Other':<30} {other_time:<15.2f} {other_time/total*100:<15.1f}%")
    print("=" * 60)
    print(f"{'TOTAL':<30} {total:<15.2f} {'100.0%':<15}")

    # Print detailed table
    logger.info("\n" + "="*100)
    logger.info("TOP 40 OPERATIONS BY CUDA TIME")
    logger.info("="*100)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=40,
        max_name_column_width=80
    ))

    # Save trace
    trace_file = "beam_decode_trace_hooks.json"
    prof.export_chrome_trace(trace_file)

    logger.info("\n" + "="*100)
    logger.info("TRACE FILE SAVED")
    logger.info("="*100)
    logger.info(f"Chrome trace: {trace_file}")
    logger.info(f"  -> Open chrome://tracing in Chrome")
    logger.info(f"  -> Click 'Load' and select {trace_file}")
    logger.info(f"  -> You'll see a hierarchical timeline!")
    logger.info("="*100)

    return generated_texts


if __name__ == "__main__":
    main()

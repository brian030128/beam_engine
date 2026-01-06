"""
Benchmark script comparing HuggingFace Transformers vs vLLM for text generation.

Model: meta-llama/Llama-3.1-8B
Dtype: FP16
Device: cuda:0

Tests:
1. Different input lengths (64, 256, 512, 1024 tokens)
2. Greedy decoding vs Beam search (beam_width=4)
3. Batch size variations

Note: HuggingFace greedy decoding uses StaticCache for optimal performance
when output length is known in advance.
"""

import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import gc
import json
from datetime import datetime


# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
DTYPE = torch.float16

# Test configurations
INPUT_LENGTHS = [64, 256, 512, 1024]
BATCH_SIZE = 5
OUTPUT_LENGTH = 128
NUM_WARMUP_RUNS = 2
NUM_BENCHMARK_RUNS = 5

# Beam search configuration
BEAM_WIDTH = 4


class DecodingStrategy(Enum):
    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    input_length: int
    output_length: int
    batch_size: int
    decoding_strategy: DecodingStrategy
    beam_width: Optional[int] = None
    
    def __str__(self) -> str:
        strategy = self.decoding_strategy.value
        if self.decoding_strategy == DecodingStrategy.BEAM_SEARCH:
            strategy += f"(width={self.beam_width})"
        return f"in={self.input_length}, out={self.output_length}, batch={self.batch_size}, {strategy}"


def generate_dummy_input(tokenizer, batch_size: int, input_length: int) -> List[str]:
    """Generate dummy input texts of approximately the specified token length."""
    base_text = "This is a sample text for benchmarking language model performance. "
    
    base_tokens = tokenizer.encode(base_text, add_special_tokens=False)
    repeats = (input_length // len(base_tokens)) + 1
    
    long_text = base_text * repeats
    tokens = tokenizer.encode(long_text, add_special_tokens=False)[:input_length]
    text = tokenizer.decode(tokens)
    
    return [text] * batch_size


def clear_gpu_memory():
    """Clear GPU memory between benchmarks."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return {"allocated_gb": allocated, "reserved_gb": reserved}
    return {"allocated_gb": 0, "reserved_gb": 0}


class HuggingFaceBenchmark:
    """Benchmark for HuggingFace Transformers with StaticCache for greedy decoding."""
    
    def __init__(self, model_name: str, device: str, dtype: torch.dtype):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading HuggingFace model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=dtype
        ).to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        
        # Cache storage for reuse across same-config runs
        self._static_cache = None
        self._cache_config = None
        
        print("HuggingFace model loaded successfully")
        print(f"GPU Memory after load: {get_gpu_memory_usage()}")
    
    def _get_static_cache(self, batch_size: int, max_cache_len: int):
        """Get or create a StaticCache for the given configuration."""
        from transformers import StaticCache
        
        cache_config = (batch_size, max_cache_len)
        
        # Reuse cache if configuration matches
        if self._static_cache is not None and self._cache_config == cache_config:
            # Reset the cache for reuse
            self._static_cache.reset()
            return self._static_cache
        
        # Create new static cache
        self._static_cache = StaticCache(
            config=self.model.config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=self.device,
            dtype=self.dtype,
        )
        self._cache_config = cache_config
        
        return self._static_cache
    
    def _generate_with_static_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Generate tokens using static KV cache for optimal greedy decoding performance.
        
        This manually implements the generation loop with a pre-allocated StaticCache,
        avoiding dynamic memory allocations during generation.
        """
        batch_size, input_length = input_ids.shape
        max_cache_len = input_length + max_new_tokens
        
        # Get or create static cache
        past_key_values = self._get_static_cache(batch_size, max_cache_len)
        
        # Prepare cache position tensor (tracks position in the sequence)
        cache_position = torch.arange(input_length, device=self.device)
        
        # Prefill: process all input tokens at once
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
        )
        
        # Get the next token (greedy)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        # Collect generated tokens
        generated_tokens = [next_token]
        
        # Update attention mask for the new token
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, 1), device=self.device, dtype=attention_mask.dtype)
        ], dim=1)
        
        # Decode: generate remaining tokens one at a time
        for i in range(max_new_tokens - 1):
            # Update cache position for the new token
            cache_position = torch.tensor([input_length + i + 1], device=self.device)
            
            # Forward pass with single token
            outputs = self.model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )
            
            # Greedy selection
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            
            # Extend attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=self.device, dtype=attention_mask.dtype)
            ], dim=1)
            
            # Optional: early stopping if all sequences hit EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break
        
        # Concatenate all generated tokens
        generated = torch.cat(generated_tokens, dim=1)
        
        # Return full sequence (input + generated)
        return torch.cat([input_ids, generated], dim=1)
    
    def generate(
        self, 
        prompts: List[str], 
        max_new_tokens: int,
        decoding_strategy: DecodingStrategy,
        beam_width: Optional[int] = None,
    ) -> List[str]:
        """Generate text using HuggingFace model."""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        with torch.no_grad():
            if decoding_strategy == DecodingStrategy.GREEDY:
                # Use optimized static cache generation for greedy decoding
                outputs = self._generate_with_static_cache(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                )
            elif decoding_strategy == DecodingStrategy.BEAM_SEARCH:
                # Beam search uses standard generate (static cache not applicable)
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "do_sample": False,
                    "num_beams": beam_width or BEAM_WIDTH,
                    "early_stopping": True,
                }
                outputs = self.model.generate(**inputs, **generation_kwargs)
            else:
                raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def benchmark(
        self,
        config: BenchmarkConfig,
        prompts: List[str],
        num_warmup: int,
        num_runs: int,
    ) -> Dict[str, Any]:
        """Run benchmark and collect metrics."""
        cache_info = " (with StaticCache)" if config.decoding_strategy == DecodingStrategy.GREEDY else ""
        print(f"  Config: {config}{cache_info}")
        
        # Warmup
        print("  Running warmup...")
        for _ in range(num_warmup):
            self.generate(
                prompts, 
                config.output_length,
                config.decoding_strategy,
                config.beam_width,
            )
        
        torch.cuda.synchronize()
        
        # Benchmark runs
        print("  Running benchmark...")
        latencies = []
        memory_usage = []
        
        for i in range(num_runs):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = self.generate(
                prompts, 
                config.output_length,
                config.decoding_strategy,
                config.beam_width,
            )
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latencies.append(end_time - start_time)
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            memory_usage.append(peak_memory)
            print(f"    Run {i+1}/{num_runs}: {latencies[-1]:.4f}s, Peak Memory: {peak_memory:.2f}GB")
        
        # Calculate metrics
        total_tokens_generated = len(prompts) * config.output_length
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        throughput = total_tokens_generated / avg_latency
        
        return {
            "framework": "HuggingFace Transformers",
            "config": str(config),
            "input_length": config.input_length,
            "output_length": config.output_length,
            "batch_size": config.batch_size,
            "decoding_strategy": config.decoding_strategy.value,
            "beam_width": config.beam_width,
            "uses_static_cache": config.decoding_strategy == DecodingStrategy.GREEDY,
            "avg_latency_s": avg_latency,
            "std_latency_s": std_latency,
            "min_latency_s": np.min(latencies),
            "max_latency_s": np.max(latencies),
            "throughput_tokens_per_s": throughput,
            "total_tokens_generated": total_tokens_generated,
            "avg_peak_memory_gb": np.mean(memory_usage),
            "latencies": latencies,
        }
    
    def cleanup(self):
        """Release model resources."""
        del self.model
        del self.tokenizer
        if self._static_cache is not None:
            del self._static_cache
        clear_gpu_memory()


class VLLMBenchmark:
    """Benchmark for vLLM."""
    
    def __init__(self, model_name: str, device: str, dtype: torch.dtype):
        from vllm import LLM
        
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        
        print(f"Loading vLLM model: {model_name}")
        
        dtype_str = "float16" if dtype == torch.float16 else "bfloat16"
        
        self.llm = LLM(
            model=model_name,
            dtype=dtype_str,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("vLLM model loaded successfully")
        print(f"GPU Memory after load: {get_gpu_memory_usage()}")
    
    def generate(
        self, 
        prompts: List[str], 
        max_new_tokens: int,
        decoding_strategy: DecodingStrategy,
        beam_width: Optional[int] = None,
    ) -> List[str]:
        """Generate text using vLLM."""
        from vllm import SamplingParams
        
        if decoding_strategy == DecodingStrategy.GREEDY:
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0,
            )
            outputs = self.llm.generate(prompts, sampling_params)
            return [output.outputs[0].text for output in outputs]
        
        elif decoding_strategy == DecodingStrategy.BEAM_SEARCH:
            # Try different vLLM beam search APIs based on version
            try:
                # vLLM >= 0.6.0: use beam_search with BeamSearchParams
                from vllm.sampling_params import BeamSearchParams
                beam_params = BeamSearchParams(
                    beam_width=beam_width or BEAM_WIDTH,
                    max_tokens=max_new_tokens,
                )
                # beam_search expects TokensPrompt dicts with "prompt_token_ids" key
                tokenizer = self.llm.get_tokenizer()
                prompt_inputs = [
                    {"prompt_token_ids": tokenizer.encode(p)} for p in prompts
                ]
                outputs = self.llm.beam_search(prompt_inputs, beam_params)
                # beam_search returns BeamSearchOutput with sequences attribute
                return [output.sequences[0].text for output in outputs]
            except (ImportError, AttributeError, TypeError) as e:
                # Fallback: use SamplingParams with use_beam_search=True (older API)
                try:
                    sampling_params = SamplingParams(
                        max_tokens=max_new_tokens,
                        temperature=0,
                        use_beam_search=True,
                        best_of=beam_width or BEAM_WIDTH,
                        n=1,
                    )
                    outputs = self.llm.generate(prompts, sampling_params)
                    return [output.outputs[0].text for output in outputs]
                except Exception:
                    # Last fallback: just use greedy and warn
                    print(f"    Warning: Beam search not available, falling back to greedy")
                    sampling_params = SamplingParams(
                        max_tokens=max_new_tokens,
                        temperature=0,
                    )
                    outputs = self.llm.generate(prompts, sampling_params)
                    return [output.outputs[0].text for output in outputs]
    
    def benchmark(
        self,
        config: BenchmarkConfig,
        prompts: List[str],
        num_warmup: int,
        num_runs: int,
    ) -> Dict[str, Any]:
        """Run benchmark and collect metrics."""
        print(f"  Config: {config}")
        
        # Warmup
        print("  Running warmup...")
        for _ in range(num_warmup):
            self.generate(
                prompts, 
                config.output_length,
                config.decoding_strategy,
                config.beam_width,
            )
        
        torch.cuda.synchronize()
        
        # Benchmark runs
        print("  Running benchmark...")
        latencies = []
        memory_usage = []
        
        for i in range(num_runs):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = self.generate(
                prompts, 
                config.output_length,
                config.decoding_strategy,
                config.beam_width,
            )
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latencies.append(end_time - start_time)
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            memory_usage.append(peak_memory)
            print(f"    Run {i+1}/{num_runs}: {latencies[-1]:.4f}s, Peak Memory: {peak_memory:.2f}GB")
        
        # Calculate metrics
        total_tokens_generated = len(prompts) * config.output_length
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        throughput = total_tokens_generated / avg_latency
        
        return {
            "framework": "vLLM",
            "config": str(config),
            "input_length": config.input_length,
            "output_length": config.output_length,
            "batch_size": config.batch_size,
            "decoding_strategy": config.decoding_strategy.value,
            "beam_width": config.beam_width,
            "avg_latency_s": avg_latency,
            "std_latency_s": std_latency,
            "min_latency_s": np.min(latencies),
            "max_latency_s": np.max(latencies),
            "throughput_tokens_per_s": throughput,
            "total_tokens_generated": total_tokens_generated,
            "avg_peak_memory_gb": np.mean(memory_usage),
            "latencies": latencies,
        }
    
    def cleanup(self):
        """Release model resources."""
        del self.llm
        clear_gpu_memory()


def print_results(results: Dict[str, Any]):
    """Pretty print benchmark results."""
    print(f"\n{'-'*60}")
    print(f"Results: {results['framework']} | {results['config']}")
    if results.get('uses_static_cache'):
        print(f"         (using StaticCache for optimized greedy decoding)")
    print(f"{'-'*60}")
    print(f"  Average latency:         {results['avg_latency_s']:.4f}s (± {results['std_latency_s']:.4f}s)")
    print(f"  Throughput:              {results['throughput_tokens_per_s']:.2f} tokens/s")
    print(f"  Peak memory:             {results['avg_peak_memory_gb']:.2f} GB")


def print_comparison_table(all_results: List[Dict[str, Any]]):
    """Print a comprehensive comparison table."""
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*100)
    
    # Group by configuration
    configs = {}
    for r in all_results:
        key = (r['input_length'], r['decoding_strategy'], r.get('beam_width'))
        if key not in configs:
            configs[key] = {}
        configs[key][r['framework']] = r
    
    # Print header
    print(f"{'Input Len':<12} {'Decoding':<20} {'Framework':<25} {'Latency (s)':<15} {'Throughput':<15} {'Memory (GB)':<12}")
    print("-"*100)
    
    for (input_len, decoding, beam), frameworks in sorted(configs.items()):
        decoding_str = decoding if decoding == 'greedy' else f'beam(w={beam})'
        
        for fw_name, r in frameworks.items():
            cache_note = " [StaticCache]" if r.get('uses_static_cache') else ""
            print(f"{input_len:<12} {decoding_str:<20} {fw_name + cache_note:<25} "
                  f"{r['avg_latency_s']:.4f}±{r['std_latency_s']:.4f}  "
                  f"{r['throughput_tokens_per_s']:<15.2f} {r['avg_peak_memory_gb']:<12.2f}")
        
        # Print speedup if both frameworks present
        if 'HuggingFace Transformers' in frameworks and 'vLLM' in frameworks:
            hf = frameworks['HuggingFace Transformers']
            vllm = frameworks['vLLM']
            speedup = hf['avg_latency_s'] / vllm['avg_latency_s']
            throughput_ratio = vllm['throughput_tokens_per_s'] / hf['throughput_tokens_per_s']
            print(f"{'':12} {'':20} {'>>> vLLM speedup:':<25} {speedup:.2f}x latency, {throughput_ratio:.2f}x throughput")
        print()


def print_summary_analysis(all_results: List[Dict[str, Any]]):
    """Print summary analysis of the benchmarks."""
    print("\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)
    
    # Separate by framework
    hf_results = [r for r in all_results if r['framework'] == 'HuggingFace Transformers']
    vllm_results = [r for r in all_results if r['framework'] == 'vLLM']
    
    # Note about StaticCache
    print("\nNote: HuggingFace greedy decoding uses StaticCache for optimal performance")
    print("      (pre-allocated KV cache avoids dynamic memory allocation overhead)")
    
    # Greedy vs Beam search comparison
    print("\n1. GREEDY vs BEAM SEARCH IMPACT")
    print("-"*40)
    
    for fw_name, fw_results in [("HuggingFace", hf_results), ("vLLM", vllm_results)]:
        greedy = [r for r in fw_results if r['decoding_strategy'] == 'greedy']
        beam = [r for r in fw_results if r['decoding_strategy'] == 'beam_search']
        
        if greedy and beam:
            avg_greedy_throughput = np.mean([r['throughput_tokens_per_s'] for r in greedy])
            avg_beam_throughput = np.mean([r['throughput_tokens_per_s'] for r in beam])
            overhead = (avg_greedy_throughput - avg_beam_throughput) / avg_greedy_throughput * 100
            print(f"  {fw_name}: Beam search reduces throughput by ~{overhead:.1f}% on average")
    
    # Input length scaling
    print("\n2. INPUT LENGTH SCALING")
    print("-"*40)
    
    for fw_name, fw_results in [("HuggingFace", hf_results), ("vLLM", vllm_results)]:
        greedy = [r for r in fw_results if r['decoding_strategy'] == 'greedy']
        if len(greedy) >= 2:
            greedy_sorted = sorted(greedy, key=lambda x: x['input_length'])
            shortest = greedy_sorted[0]
            longest = greedy_sorted[-1]
            latency_increase = longest['avg_latency_s'] / shortest['avg_latency_s']
            print(f"  {fw_name}: {shortest['input_length']} -> {longest['input_length']} tokens: "
                  f"{latency_increase:.2f}x latency increase")
    
    # Overall speedup
    print("\n3. OVERALL vLLM SPEEDUP vs HuggingFace (with StaticCache)")
    print("-"*40)
    
    speedups = []
    for hf_r in hf_results:
        matching = [v for v in vllm_results 
                   if v['input_length'] == hf_r['input_length'] 
                   and v['decoding_strategy'] == hf_r['decoding_strategy']]
        if matching:
            speedups.append(hf_r['avg_latency_s'] / matching[0]['avg_latency_s'])
    
    if speedups:
        print(f"  Average speedup: {np.mean(speedups):.2f}x")
        print(f"  Min speedup: {np.min(speedups):.2f}x")
        print(f"  Max speedup: {np.max(speedups):.2f}x")


def save_results(all_results: List[Dict[str, Any]], filename: str):
    """Save results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj
    
    serializable_results = []
    for r in all_results:
        serializable_results.append({k: convert(v) for k, v in r.items()})
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "notes": "HuggingFace greedy decoding uses StaticCache for optimal performance",
        "results": serializable_results,
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filename}")


def main():
    """Main benchmark function."""
    print("="*80)
    print("Text Generation Benchmark: HuggingFace vs vLLM")
    print("="*80)
    print(f"Model:         {MODEL_NAME}")
    print(f"Device:        {DEVICE}")
    print(f"Dtype:         {DTYPE}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Input lengths: {INPUT_LENGTHS}")
    print(f"Output length: {OUTPUT_LENGTH} tokens")
    print(f"Beam width:    {BEAM_WIDTH}")
    print(f"Warmup runs:   {NUM_WARMUP_RUNS}")
    print(f"Bench runs:    {NUM_BENCHMARK_RUNS}")
    print()
    print("Note: HuggingFace greedy uses StaticCache (pre-allocated KV cache)")
    print("="*80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Build test configurations
    configs = []
    for input_length in INPUT_LENGTHS:
        # Greedy decoding
        configs.append(BenchmarkConfig(
            input_length=input_length,
            output_length=OUTPUT_LENGTH,
            batch_size=BATCH_SIZE,
            decoding_strategy=DecodingStrategy.GREEDY,
        ))
        # Beam search
        configs.append(BenchmarkConfig(
            input_length=input_length,
            output_length=OUTPUT_LENGTH,
            batch_size=BATCH_SIZE,
            decoding_strategy=DecodingStrategy.BEAM_SEARCH,
            beam_width=BEAM_WIDTH,
        ))
    
    all_results = []
    
    # ==========================================
    # Benchmark HuggingFace
    # ==========================================
    print("\n" + "="*80)
    print("BENCHMARKING HUGGINGFACE TRANSFORMERS")
    print("(Greedy decoding uses StaticCache for optimal performance)")
    print("="*80)
    
    hf_benchmark = HuggingFaceBenchmark(MODEL_NAME, DEVICE, DTYPE)
    
    for config in configs:
        prompts = generate_dummy_input(hf_benchmark.tokenizer, config.batch_size, config.input_length)
        
        result = hf_benchmark.benchmark(
            config=config,
            prompts=prompts,
            num_warmup=NUM_WARMUP_RUNS,
            num_runs=NUM_BENCHMARK_RUNS,
        )
        print_results(result)
        all_results.append(result)
    
    hf_benchmark.cleanup()
    
    # ==========================================
    # Benchmark vLLM
    # ==========================================
    print("\n" + "="*80)
    print("BENCHMARKING vLLM")
    print("="*80)
    
    vllm_benchmark = VLLMBenchmark(MODEL_NAME, DEVICE, DTYPE)
    
    for config in configs:
        prompts = generate_dummy_input(vllm_benchmark.tokenizer, config.batch_size, config.input_length)
        
        result = vllm_benchmark.benchmark(
            config=config,
            prompts=prompts,
            num_warmup=NUM_WARMUP_RUNS,
            num_runs=NUM_BENCHMARK_RUNS,
        )
        print_results(result)
        all_results.append(result)
    
    vllm_benchmark.cleanup()
    
    # ==========================================
    # Print comparisons and analysis
    # ==========================================
    print_comparison_table(all_results)
    print_summary_analysis(all_results)
    
    # Save results
    save_results(all_results, "benchmark_results.json")
    
    return all_results


if __name__ == "__main__":
    main()
"""
Benchmark script comparing HuggingFace Transformers vs vLLM for text generation.

Model: meta-llama/Llama-3.1-8B
Dtype: FP16
Device: cuda:0
Batch size: 5
Input length: 100 tokens
"""

import time
import torch
import numpy as np
from typing import List, Dict, Any
import gc


# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"
DTYPE = torch.float16
BATCH_SIZE = 5
INPUT_LENGTH = 100
OUTPUT_LENGTH = 128  # Generated tokens
NUM_WARMUP_RUNS = 2
NUM_BENCHMARK_RUNS = 5


def generate_dummy_input(tokenizer, batch_size: int, input_length: int) -> List[str]:
    """Generate dummy input texts of approximately the specified token length."""
    # Create a base text and repeat to reach desired length
    base_text = "This is a sample text for benchmarking language model performance. "
    
    # Tokenize base text to know its length
    base_tokens = tokenizer.encode(base_text, add_special_tokens=False)
    repeats = (input_length // len(base_tokens)) + 1
    
    long_text = base_text * repeats
    # Truncate to exact token length
    tokens = tokenizer.encode(long_text, add_special_tokens=False)[:input_length]
    text = tokenizer.decode(tokens)
    
    return [text] * batch_size


def clear_gpu_memory():
    """Clear GPU memory between benchmarks."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


class HuggingFaceBenchmark:
    """Benchmark for HuggingFace Transformers."""
    
    def __init__(self, model_name: str, device: str, dtype: torch.dtype):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading HuggingFace model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model =  AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        self.model.eval()
        self.device = device
        print("HuggingFace model loaded successfully")
    
    def generate(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """Generate text using HuggingFace model."""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for reproducibility
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def benchmark(
        self,
        prompts: List[str],
        max_new_tokens: int,
        num_warmup: int,
        num_runs: int,
    ) -> Dict[str, Any]:
        """Run benchmark and collect metrics."""
        # Warmup
        print("  Running warmup...")
        for _ in range(num_warmup):
            self.generate(prompts, max_new_tokens)
        
        torch.cuda.synchronize()
        
        # Benchmark runs
        print("  Running benchmark...")
        latencies = []
        for i in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = self.generate(prompts, max_new_tokens)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latencies.append(end_time - start_time)
            print(f"    Run {i+1}/{num_runs}: {latencies[-1]:.4f}s")
        
        # Calculate metrics
        total_tokens_generated = len(prompts) * max_new_tokens
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        throughput = total_tokens_generated / avg_latency
        
        return {
            "framework": "HuggingFace Transformers",
            "avg_latency_s": avg_latency,
            "std_latency_s": std_latency,
            "min_latency_s": np.min(latencies),
            "max_latency_s": np.max(latencies),
            "throughput_tokens_per_s": throughput,
            "total_tokens_generated": total_tokens_generated,
            "batch_size": len(prompts),
        }
    
    def cleanup(self):
        """Release model resources."""
        del self.model
        del self.tokenizer
        clear_gpu_memory()


class VLLMBenchmark:
    """Benchmark for vLLM."""
    
    def __init__(self, model_name: str, device: str, dtype: torch.dtype):
        from vllm import LLM, SamplingParams
        
        # Extract GPU index from device string
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        
        print(f"Loading vLLM model: {model_name}")
        
        # Map torch dtype to vLLM dtype string
        dtype_str = "float16" if dtype == torch.float16 else "bfloat16"
        
        self.llm = LLM(
            model=model_name,
            dtype=dtype_str,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("vLLM model loaded successfully")
    
    def generate(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """Generate text using vLLM."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0,  # Greedy decoding
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def benchmark(
        self,
        prompts: List[str],
        max_new_tokens: int,
        num_warmup: int,
        num_runs: int,
    ) -> Dict[str, Any]:
        """Run benchmark and collect metrics."""
        # Warmup
        print("  Running warmup...")
        for _ in range(num_warmup):
            self.generate(prompts, max_new_tokens)
        
        torch.cuda.synchronize()
        
        # Benchmark runs
        print("  Running benchmark...")
        latencies = []
        for i in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = self.generate(prompts, max_new_tokens)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latencies.append(end_time - start_time)
            print(f"    Run {i+1}/{num_runs}: {latencies[-1]:.4f}s")
        
        # Calculate metrics
        total_tokens_generated = len(prompts) * max_new_tokens
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        throughput = total_tokens_generated / avg_latency
        
        return {
            "framework": "vLLM",
            "avg_latency_s": avg_latency,
            "std_latency_s": std_latency,
            "min_latency_s": np.min(latencies),
            "max_latency_s": np.max(latencies),
            "throughput_tokens_per_s": throughput,
            "total_tokens_generated": total_tokens_generated,
            "batch_size": len(prompts),
        }
    
    def cleanup(self):
        """Release model resources."""
        del self.llm
        clear_gpu_memory()


def print_results(results: Dict[str, Any]):
    """Pretty print benchmark results."""
    print(f"\n{'='*60}")
    print(f"Results for: {results['framework']}")
    print(f"{'='*60}")
    print(f"  Batch size:              {results['batch_size']}")
    print(f"  Total tokens generated:  {results['total_tokens_generated']}")
    print(f"  Average latency:         {results['avg_latency_s']:.4f}s (± {results['std_latency_s']:.4f}s)")
    print(f"  Min latency:             {results['min_latency_s']:.4f}s")
    print(f"  Max latency:             {results['max_latency_s']:.4f}s")
    print(f"  Throughput:              {results['throughput_tokens_per_s']:.2f} tokens/s")


def print_comparison(hf_results: Dict[str, Any], vllm_results: Dict[str, Any]):
    """Print comparison between frameworks."""
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    speedup = hf_results['avg_latency_s'] / vllm_results['avg_latency_s']
    throughput_ratio = vllm_results['throughput_tokens_per_s'] / hf_results['throughput_tokens_per_s']
    
    print(f"  vLLM speedup over HuggingFace: {speedup:.2f}x")
    print(f"  vLLM throughput improvement:   {throughput_ratio:.2f}x")
    print(f"\n  HuggingFace throughput: {hf_results['throughput_tokens_per_s']:.2f} tokens/s")
    print(f"  vLLM throughput:        {vllm_results['throughput_tokens_per_s']:.2f} tokens/s")


def main():
    """Main benchmark function."""
    print("="*60)
    print("Text Generation Benchmark: HuggingFace vs vLLM")
    print("="*60)
    print(f"Model:        {MODEL_NAME}")
    print(f"Device:       {DEVICE}")
    print(f"Dtype:        {DTYPE}")
    print(f"Batch size:   {BATCH_SIZE}")
    print(f"Input length: {INPUT_LENGTH} tokens")
    print(f"Output length:{OUTPUT_LENGTH} tokens")
    print(f"Warmup runs:  {NUM_WARMUP_RUNS}")
    print(f"Bench runs:   {NUM_BENCHMARK_RUNS}")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    results = {}
    
    # Benchmark HuggingFace
    print("\n" + "="*60)
    print("Benchmarking HuggingFace Transformers...")
    print("="*60)
    
    hf_benchmark = HuggingFaceBenchmark(MODEL_NAME, DEVICE, DTYPE)
    prompts = generate_dummy_input(hf_benchmark.tokenizer, BATCH_SIZE, INPUT_LENGTH)
    
    hf_results = hf_benchmark.benchmark(
        prompts=prompts,
        max_new_tokens=OUTPUT_LENGTH,
        num_warmup=NUM_WARMUP_RUNS,
        num_runs=NUM_BENCHMARK_RUNS,
    )
    hf_benchmark.cleanup()
    print_results(hf_results)
    results['huggingface'] = hf_results
    
    # Benchmark vLLM
    print("\n" + "="*60)
    print("Benchmarking vLLM...")
    print("="*60)
    
    vllm_benchmark = VLLMBenchmark(MODEL_NAME, DEVICE, DTYPE)
    
    vllm_results = vllm_benchmark.benchmark(
        prompts=prompts,
        max_new_tokens=OUTPUT_LENGTH,
        num_warmup=NUM_WARMUP_RUNS,
        num_runs=NUM_BENCHMARK_RUNS,
    )
    vllm_benchmark.cleanup()
    print_results(vllm_results)
    results['vllm'] = vllm_results
    
    # Print comparison
    print_comparison(hf_results, vllm_results)
    
    return results


if __name__ == "__main__":
    main()

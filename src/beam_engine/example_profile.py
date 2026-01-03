"""
Example: How to profile your decode forward pass.

This shows how to use PyTorch's profiler to automatically identify bottlenecks.
"""

import torch
from torch.profiler import profile, ProfilerActivity, record_function

# Example usage with your model
def profile_your_decode():
    """
    Template for profiling your actual decode forward pass.
    Modify this to match your actual setup.
    """

    # TODO: Replace with your actual model loading
    # model = YourModel.from_pretrained(...)
    # page_table = PageTable(...)
    # model.eval()
    # model = model.cuda()

    # TODO: Prepare your inputs for decode
    # This is just a template - adjust to your actual input format
    # inputs = {
    #     'input_ids': ...,
    #     'attention_mode': AttentionMode.DECODE,
    #     'page_table': page_table,
    #     'cascade_qo_indptr_arr': ...,
    #     'cascade_kv_indptr_arr': ...,
    #     'cascade_kv_indices_arr': ...,
    #     'cascade_kv_last_page_len_arr': ...,
    #     'cascade_write_page_indices': ...,
    #     'cascade_write_positions': ...,
    # }

    print("Starting profiling...")
    print("Make sure CUDA is available:", torch.cuda.is_available())

    # Profile with PyTorch profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("warmup"):
            for _ in range(3):
                with torch.no_grad():
                    pass  # model(**inputs)

        with record_function("decode_forward"):
            for _ in range(10):
                with torch.no_grad():
                    pass  # outputs = model(**inputs)
                prof.step()

    # Print results - automatically shows all operations
    print("\n" + "="*100)
    print("TOP OPERATIONS BY CUDA TIME")
    print("="*100)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    print("\n" + "="*100)
    print("TOP OPERATIONS BY CPU TIME")
    print("="*100)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    # Export Chrome trace for visualization
    prof.export_chrome_trace("decode_trace.json")
    print("\n✓ Trace saved to decode_trace.json")
    print("  View it at chrome://tracing in Chrome browser")


# Quick inline profiling example
def quick_profile_example():
    """
    Quickest way to profile - just wrap your code.
    """
    import torch
    from torch.profiler import profile, ProfilerActivity

    # Your model and inputs here
    # model = ...
    # inputs = ...

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        # Your forward pass here
        # model(**inputs)
        pass

    # Print top 20 operations
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    print(__doc__)
    print("\nHow to use:")
    print("1. Uncomment and fill in your model/data loading code")
    print("2. Run: python example_profile.py")
    print("3. Check the printed table to see where time is spent")
    print("\nThe profiler automatically captures:")
    print("  - All CUDA kernels (attention, matmuls, etc.)")
    print("  - All CPU operations")
    print("  - Memory allocations")
    print("  - FlashInfer kernels")
    print("  - Everything else!")
    print("\nNo need to manually instrument your code!")

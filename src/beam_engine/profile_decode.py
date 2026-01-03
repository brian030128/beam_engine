"""
Simple script to profile model decoding and identify bottlenecks.

This uses PyTorch's built-in profiler which automatically captures all operations
without needing to instrument the code.
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity


def profile_model_forward(
    model,
    input_data,
    profile_dir="./profiler_logs",
    warmup_steps=2,
    active_steps=5,
    save_chrome_trace=True
):
    """
    Profile a model's forward pass and generate detailed performance report.

    Args:
        model: The model to profile
        input_data: Dictionary of inputs to pass to model.forward()
        profile_dir: Directory to save profiling results
        warmup_steps: Number of warmup iterations
        active_steps: Number of active profiling iterations
        save_chrome_trace: Whether to save Chrome trace (viewable in chrome://tracing)

    Returns:
        Profiler object with results
    """
    print("Starting profiling...")
    print(f"Warmup: {warmup_steps} steps, Active: {active_steps} steps")

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        # Warmup
        for _ in range(warmup_steps):
            with torch.no_grad():
                model(**input_data)

        # Active profiling
        for _ in range(active_steps):
            with torch.no_grad():
                with record_function("model_forward"):
                    model(**input_data)
            prof.step()

    # Print summary reports
    print("\n" + "="*100)
    print("TOP 20 OPERATIONS BY CUDA TIME")
    print("="*100)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
        header="Top 20 operations by CUDA time"
    ))

    print("\n" + "="*100)
    print("TOP 20 OPERATIONS BY CPU TIME")
    print("="*100)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=20,
        header="Top 20 operations by CPU time"
    ))

    print("\n" + "="*100)
    print("OPERATIONS GROUPED BY INPUT SHAPE")
    print("="*100)
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total",
        row_limit=20
    ))

    if save_chrome_trace:
        trace_file = f"{profile_dir}/trace.json"
        prof.export_chrome_trace(trace_file)
        print(f"\nChrome trace saved to: {trace_file}")
        print(f"View it at: chrome://tracing")

    # Save as JSON for programmatic analysis
    import json
    stats_file = f"{profile_dir}/stats.json"

    # Extract key averages as structured data
    stats = []
    for event in prof.key_averages():
        stats.append({
            "name": event.key,
            "cpu_time": event.cpu_time,
            "cuda_time": event.cuda_time,
            "cpu_time_total": event.cpu_time_total,
            "cuda_time_total": event.cuda_time_total,
            "count": event.count,
            "input_shapes": str(event.input_shapes) if hasattr(event, 'input_shapes') else None,
        })

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to: {stats_file}")

    return prof


# Example usage function
def example_usage():
    """
    Example of how to use this profiler with your model.
    """
    # Assuming you have your model and inputs ready
    # from your_model import model
    # from your_data import prepare_decode_inputs

    # # Prepare your decode inputs
    # input_data = prepare_decode_inputs(
    #     num_candidates=8,
    #     attention_mode=AttentionMode.DECODE,
    #     page_table=page_table,
    #     # ... other parameters
    # )

    # # Profile the forward pass
    # prof = profile_model_forward(
    #     model=model,
    #     input_data=input_data,
    #     profile_dir="./decode_profiling",
    #     warmup_steps=2,
    #     active_steps=5
    # )

    pass


if __name__ == "__main__":
    print(__doc__)
    print("\nTo use this profiler:")
    print("1. Import your model and prepare inputs")
    print("2. Call profile_model_forward(model, input_data)")
    print("3. Check the printed reports and saved trace files")
    print("\nSee example_usage() function for a template.")

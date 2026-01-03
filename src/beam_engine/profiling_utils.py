"""
Profiling utilities for measuring performance bottlenecks.

Quick usage:
    from profiling_utils import auto_profile

    # Wrap your function
    @auto_profile
    def my_function():
        ...

    # Or use inline:
    with auto_profile("section_name"):
        ...
"""

import torch
import time
from contextlib import contextmanager
from typing import Optional, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class ProfilerStats:
    """Simple profiler for tracking execution times."""

    def __init__(self):
        self.times = {}
        self.counts = {}

    def record(self, name: str, elapsed_ms: float):
        if name not in self.times:
            self.times[name] = 0.0
            self.counts[name] = 0
        self.times[name] += elapsed_ms
        self.counts[name] += 1

    def report(self):
        """Print profiling report sorted by total time."""
        print("\n" + "="*80)
        print("PROFILING REPORT")
        print("="*80)
        print(f"{'Section':<40} {'Count':>8} {'Total (ms)':>12} {'Avg (ms)':>12}")
        print("-"*80)

        # Sort by total time descending
        sorted_items = sorted(self.times.items(), key=lambda x: x[1], reverse=True)

        total_time = sum(self.times.values())

        for name, total_ms in sorted_items:
            count = self.counts[name]
            avg_ms = total_ms / count if count > 0 else 0
            pct = (total_ms / total_time * 100) if total_time > 0 else 0
            print(f"{name:<40} {count:>8} {total_ms:>12.3f} {avg_ms:>12.3f}  ({pct:>5.1f}%)")

        print("-"*80)
        print(f"{'TOTAL':<40} {' '*8} {total_time:>12.3f}")
        print("="*80 + "\n")

    def reset(self):
        """Reset all statistics."""
        self.times.clear()
        self.counts.clear()


# Global profiler instance
_global_profiler = ProfilerStats()


@contextmanager
def profile_section(name: str, enabled: bool = True, use_cuda: bool = True):
    """
    Context manager for profiling a code section.

    Usage:
        with profile_section("my_section"):
            # code to profile
            ...

    Args:
        name: Name of the section to profile
        enabled: Whether profiling is enabled
        use_cuda: Whether to use CUDA events for GPU timing (more accurate)
    """
    if not enabled:
        yield
        return

    if use_cuda and torch.cuda.is_available():
        # Use CUDA events for accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        yield
        end_event.record()

        # Synchronize to get accurate timing
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        # Use CPU timing
        start_time = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start_time) * 1000

    _global_profiler.record(name, elapsed_ms)


def get_profiler() -> ProfilerStats:
    """Get the global profiler instance."""
    return _global_profiler


def print_profile_report():
    """Print the profiling report."""
    _global_profiler.report()


def reset_profiler():
    """Reset the profiler statistics."""
    _global_profiler.reset()


# PyTorch Profiler wrapper for more detailed analysis
@contextmanager
def torch_profiler(
    profile_dir: str = "./profiler_logs",
    with_stack: bool = True,
    with_flops: bool = False,
    record_shapes: bool = True
):
    """
    Context manager for PyTorch's built-in profiler.

    This provides detailed profiling including GPU kernels, memory usage, etc.
    Results can be visualized with TensorBoard or Chrome tracing.

    Usage:
        with torch_profiler(profile_dir="./logs"):
            model(inputs)

    Then view results with:
        tensorboard --logdir=./logs

    Args:
        profile_dir: Directory to save profiling results
        with_stack: Record Python stack traces
        with_flops: Record FLOPs (requires torchprof)
        record_shapes: Record tensor shapes
    """
    from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler(profile_dir),
        record_shapes=record_shapes,
        with_stack=with_stack,
        with_flops=with_flops,
    ) as prof:
        yield prof

    # Print summary
    print("\n" + "="*80)
    print("PyTorch Profiler Summary (Top 10 by CUDA time)")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(f"\nDetailed trace saved to: {profile_dir}")
    print("View with: tensorboard --logdir=" + profile_dir)
    print("="*80 + "\n")


# Simple auto-profile decorator
def auto_profile(func_or_name=None, use_cuda=True):
    """
    Decorator or context manager for automatic profiling.

    Usage as decorator:
        @auto_profile
        def my_function():
            ...

    Usage as context manager:
        with auto_profile("section_name"):
            ...
    """
    # Used as context manager
    if isinstance(func_or_name, str):
        return profile_section(func_or_name, enabled=True, use_cuda=use_cuda)

    # Used as decorator
    if callable(func_or_name):
        func = func_or_name

        @wraps(func)
        def wrapper(*args, **kwargs):
            with profile_section(func.__name__, enabled=True, use_cuda=use_cuda):
                return func(*args, **kwargs)
        return wrapper

    # Used as @auto_profile() with parens
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with profile_section(func.__name__, enabled=True, use_cuda=use_cuda):
                return func(*args, **kwargs)
        return wrapper
    return decorator

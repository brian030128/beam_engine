"""Tensor-parallel support for beam_engine.

Mirrors the minimum slice of SGLang's TP design we need to run a single
beam-search instance across multiple GPUs (one TP group, NCCL backend,
column/row-parallel linears, all-reduce after each o_proj / down_proj).

Public API:
  * ``init_tp(...)`` — initialize the TP process group.
  * ``get_tp_world_size()`` / ``get_tp_rank()`` / ``get_tp_group()``.
  * ``ColumnParallelLinear`` / ``RowParallelLinear`` /
    ``QKVParallelLinear`` / ``MergedColumnParallelLinear`` — shardable
    drop-in replacements for ``nn.Linear`` in the Llama model.
"""

from .parallel_state import (
    init_tp,
    destroy_tp,
    get_tp_world_size,
    get_tp_rank,
    get_tp_group,
    is_tp_initialized,
    tp_all_reduce,
)
from .linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)

__all__ = [
    "init_tp",
    "destroy_tp",
    "get_tp_world_size",
    "get_tp_rank",
    "get_tp_group",
    "is_tp_initialized",
    "tp_all_reduce",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "QKVParallelLinear",
    "MergedColumnParallelLinear",
]

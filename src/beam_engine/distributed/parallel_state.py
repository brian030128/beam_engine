"""Tensor-parallel process group state.

A single global TP group with NCCL backend, modeled after the bits of
``sglang/srt/distributed/parallel_state.py`` we actually need. Beam_engine
runs one beam-search instance, so the TP group is just the whole world —
no PP / DP / EP slicing here.

When ``init_tp`` is never called (the default, single-GPU path), all
accessors report world_size=1 and ``tp_all_reduce`` is a no-op. This lets
the same Llama model code drive both single-GPU and multi-GPU runs.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


_TP_GROUP: Optional[dist.ProcessGroup] = None
_TP_WORLD_SIZE: int = 1
_TP_RANK: int = 0
_LOCAL_RANK: int = 0


def init_tp(
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    local_rank: Optional[int] = None,
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> None:
    """Initialize the global TP process group.

    ``world_size`` / ``rank`` / ``local_rank`` default to the ``WORLD_SIZE``
    / ``RANK`` / ``LOCAL_RANK`` env vars set by ``torchrun``. ``init_method``
    defaults to ``"env://"`` (also torchrun's convention — uses
    ``MASTER_ADDR`` / ``MASTER_PORT``).

    Pins this process to ``cuda:local_rank`` before constructing the group.
    """
    global _TP_GROUP, _TP_WORLD_SIZE, _TP_RANK, _LOCAL_RANK

    if _TP_GROUP is not None:
        return

    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if world_size == 1:
        # Single-GPU run; no need for a process group. Just record the rank.
        _TP_WORLD_SIZE = 1
        _TP_RANK = 0
        _LOCAL_RANK = local_rank
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method or "env://",
            world_size=world_size,
            rank=rank,
        )

    _TP_GROUP = dist.group.WORLD  # one global TP group; matches SGLang's _TP
    _TP_WORLD_SIZE = world_size
    _TP_RANK = rank
    _LOCAL_RANK = local_rank


def destroy_tp() -> None:
    """Tear down the TP group (mostly used by tests)."""
    global _TP_GROUP, _TP_WORLD_SIZE, _TP_RANK
    if dist.is_initialized():
        dist.destroy_process_group()
    _TP_GROUP = None
    _TP_WORLD_SIZE = 1
    _TP_RANK = 0


def is_tp_initialized() -> bool:
    return _TP_WORLD_SIZE > 1


def get_tp_world_size() -> int:
    return _TP_WORLD_SIZE


def get_tp_rank() -> int:
    return _TP_RANK


def get_local_rank() -> int:
    return _LOCAL_RANK


def get_tp_group() -> Optional[dist.ProcessGroup]:
    return _TP_GROUP


def tp_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """In-place all-reduce (sum) across the TP group. No-op when TP=1.

    Matches the contract of SGLang's ``tensor_model_parallel_all_reduce`` —
    used at the end of every ``RowParallelLinear.forward`` so the partial
    sums each rank computed (each holding its slice of the input dimension)
    add up to the full activation that all ranks then carry forward.
    """
    if _TP_WORLD_SIZE == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=_TP_GROUP)
    return tensor

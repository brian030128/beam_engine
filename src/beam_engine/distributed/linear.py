"""Tensor-parallel linear layers.

Mirrors SGLang's ``layers/linear.py`` with the minimal slice needed by the
Llama model:

  * ``ColumnParallelLinear``       — output dim sharded; per-rank weight is
    ``[out/tp, in]``; output is kept parallel (no gather).
  * ``MergedColumnParallelLinear`` — same, but the output is the
    concatenation of two independently column-sharded chunks (used to
    represent ``gate_up_proj`` as a single GEMM).
  * ``QKVParallelLinear``          — column-parallel with separate splits
    for Q / K / V (so the GQA case ``num_kv_heads < num_q_heads`` lines up
    correctly per rank).
  * ``RowParallelLinear``          — input dim sharded; per-rank weight is
    ``[out, in/tp]``; output is all-reduced.

Each layer carries a ``weight_loader`` that knows how to take the full
fused HF tensor and slice this rank's piece out of it. The Llama model's
``_remap_state_dict`` already fuses ``q_proj/k_proj/v_proj`` into
``qkv_proj`` and ``gate_proj/up_proj`` into ``gate_up_proj`` on CPU, so
by the time we load into the model the tensor shapes match those
loaders.

When ``tp_size == 1`` every loader/forward path is the trivial identity:
the layers degenerate to plain ``nn.Linear`` and the call sites match the
old beam_engine code byte-for-byte.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .parallel_state import get_tp_rank, get_tp_world_size, tp_all_reduce


def _shard_along_dim(
    full: torch.Tensor, dim: int, tp_size: int, tp_rank: int
) -> torch.Tensor:
    """Slice this rank's contiguous chunk along ``dim``.

    ``full.shape[dim]`` must be divisible by ``tp_size``. Returns a view —
    callers usually ``.clone()`` it (or rely on ``copy_``) before assigning
    into the per-rank parameter.
    """
    size = full.shape[dim]
    assert size % tp_size == 0, (
        f"dim {dim} size {size} not divisible by tp_size {tp_size}"
    )
    shard = size // tp_size
    start = tp_rank * shard
    return full.narrow(dim, start, shard)


class ColumnParallelLinear(nn.Module):
    """``y = x @ W.T (+ b)``, with the output dim split across ranks.

    Per-rank weight is ``[out_features / tp, in_features]``; the local
    output has the per-rank slice of the full output. Callers (e.g.
    ``RowParallelLinear`` consuming the output) are responsible for
    keeping the activation parallel across the TP group — there is no
    all-gather here.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        assert out_features % tp_size == 0, (
            f"ColumnParallelLinear: out_features={out_features} not divisible "
            f"by tp_size={tp_size}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_partition = out_features // tp_size
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, full: torch.Tensor) -> None:
        """Copy this rank's slice of the full HF weight into ``self.weight``."""
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()
        shard = _shard_along_dim(full, dim=0, tp_size=tp_size, tp_rank=tp_rank)
        with torch.no_grad():
            self.weight.copy_(shard)

    def bias_loader(self, full: torch.Tensor) -> None:
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()
        shard = _shard_along_dim(full, dim=0, tp_size=tp_size, tp_rank=tp_rank)
        with torch.no_grad():
            self.bias.copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(nn.Module):
    """Column-parallel linear whose output is the concat of N chunks that
    are *each* sharded independently across ranks.

    Used for ``gate_up_proj``: gate and up each have shape ``[I, hidden]``
    and the fused output is ``[2*I, hidden]``. With ``tp_size=2`` the
    fused output naively split is ``[gate, up_top_half, up_bot_half]`` —
    wrong. We want each rank to see ``[gate_shard, up_shard]`` so that
    ``silu_and_mul`` (which splits its input at the midpoint) does the
    right per-element op. This class implements that "shard then
    concatenate" layout.
    """

    def __init__(
        self,
        in_features: int,
        output_sizes: Sequence[int],
        bias: bool = False,
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        for o in output_sizes:
            assert o % tp_size == 0, (
                f"MergedColumnParallelLinear: chunk size {o} not divisible "
                f"by tp_size {tp_size}"
            )
        self.in_features = in_features
        self.output_sizes = list(output_sizes)
        self.output_sizes_per_partition = [o // tp_size for o in output_sizes]
        self.out_features_per_partition = sum(self.output_sizes_per_partition)
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, full: torch.Tensor) -> None:
        """``full`` is the fused ``[sum(output_sizes), in]`` tensor in the
        same order as ``output_sizes``. We slice each chunk independently
        and write them back-to-back into ``self.weight``."""
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()
        out_offset = 0
        local_offset = 0
        with torch.no_grad():
            for chunk_size in self.output_sizes:
                chunk = full[out_offset : out_offset + chunk_size]
                shard_size = chunk_size // tp_size
                start = tp_rank * shard_size
                self.weight[local_offset : local_offset + shard_size].copy_(
                    chunk[start : start + shard_size]
                )
                out_offset += chunk_size
                local_offset += shard_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class QKVParallelLinear(nn.Module):
    """Column-parallel projection that produces ``[q | k | v]``.

    Q is sharded by num_attention_heads, K and V by num_kv_heads. For GQA
    (num_kv_heads < num_attention_heads), the K/V shard is smaller than
    the Q shard but the same arrangement applies — each rank holds a
    contiguous chunk of heads of each.

    The full HF tensor is assumed to already be fused (the existing
    ``_remap_state_dict`` does this on CPU) with layout
    ``[q_size + kv_size + kv_size, hidden]``.

    This implementation requires ``num_kv_heads % tp_size == 0`` — i.e.
    no KV-head replication. Llama-3.1-8B has 8 KV heads, so tp_size up to
    8 works; that covers all the configurations beam_engine is benched
    on. A future addition could replicate KV heads for tp_size > num_kv_heads.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        bias: bool = False,
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        assert num_attention_heads % tp_size == 0, (
            f"QKVParallelLinear: num_attention_heads={num_attention_heads} "
            f"not divisible by tp_size={tp_size}"
        )
        assert num_kv_heads % tp_size == 0, (
            f"QKVParallelLinear: num_kv_heads={num_kv_heads} not divisible "
            f"by tp_size={tp_size}; KV replication is not yet supported"
        )

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_q_heads_total = num_attention_heads
        self.num_kv_heads_total = num_kv_heads
        self.num_q_heads_per_partition = num_attention_heads // tp_size
        self.num_kv_heads_per_partition = num_kv_heads // tp_size

        self.q_size_total = num_attention_heads * head_dim
        self.kv_size_total = num_kv_heads * head_dim
        self.q_size = self.num_q_heads_per_partition * head_dim
        self.kv_size = self.num_kv_heads_per_partition * head_dim

        self.out_features_per_partition = self.q_size + 2 * self.kv_size
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, hidden_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, full: torch.Tensor) -> None:
        """``full`` shape: ``[q_total + 2*kv_total, hidden]`` (already fused
        by ``_remap_state_dict``). Each section is sharded along the
        head-count axis with this rank's slice."""
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()
        q_full = full[: self.q_size_total]
        k_full = full[self.q_size_total : self.q_size_total + self.kv_size_total]
        v_full = full[self.q_size_total + self.kv_size_total :]

        def _slice(t: torch.Tensor, shard_size: int) -> torch.Tensor:
            start = tp_rank * shard_size
            return t[start : start + shard_size]

        with torch.no_grad():
            self.weight[: self.q_size].copy_(_slice(q_full, self.q_size))
            self.weight[self.q_size : self.q_size + self.kv_size].copy_(
                _slice(k_full, self.kv_size)
            )
            self.weight[self.q_size + self.kv_size :].copy_(
                _slice(v_full, self.kv_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """``y = x_parallel @ W.T``, with the *input* dim sharded across ranks.

    Each rank computes a partial sum that includes only its slice of the
    input dim's contribution. The all-reduce at the end of forward sums
    those partials so every rank ends up with the full activation.

    Bias is held in full size but only added on rank 0 to avoid the bias
    being summed ``tp_size`` times in the all-reduce.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        assert in_features % tp_size == 0, (
            f"RowParallelLinear: in_features={in_features} not divisible "
            f"by tp_size={tp_size}"
        )
        self.in_features = in_features
        self.in_features_per_partition = in_features // tp_size
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, full: torch.Tensor) -> None:
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank()
        shard = _shard_along_dim(full, dim=1, tp_size=tp_size, tp_rank=tp_rank)
        with torch.no_grad():
            self.weight.copy_(shard)

    def bias_loader(self, full: torch.Tensor) -> None:
        with torch.no_grad():
            self.bias.copy_(full)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No bias inside the matmul — we add it on rank 0 only after the
        # all-reduce, mirroring SGLang's RowParallelLinear (see
        # layers/linear.py L1379-L1398).
        out = torch.nn.functional.linear(x, self.weight, bias=None)
        out = tp_all_reduce(out)
        if self.bias is not None and get_tp_rank() == 0:
            out = out + self.bias
        return out

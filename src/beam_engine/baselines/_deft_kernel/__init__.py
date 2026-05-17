"""Vendored DeFT (MLSys'25) Triton attention kernel.

The single relevant entry point is ``tree_attention.tree_attention_fwd``
(split-by-node two-stage SplitK kernel). It's self-contained — only depends
on ``torch`` + ``triton`` — so we copy the source verbatim from
``3rdparty/DeFT/DeFT/deft/layers/attention/tree_attention.py`` rather than
installing DeFT (its pyproject pins torch==2.5.1 / python>=3.12, incompatible
with our env).

Kernel contract (per decode step):
  Q                  : (query_num, num_heads, head_dim)
  K_buffer/V_buffer  : (total_slots, num_kv_heads, head_dim) — slot-indexed
                       flat view of our PageTable (kv[i].view(...))
  KV_indices         : (sum_node_slots,) int32 — concatenated per-node slots
  KV_indices_offset  : (kv_num,) int32 — per-node start into KV_indices
  KV_len             : (kv_num,) int32 — per-node slot count
  KVMapQ_List        : (sum_node_queries,) int32 — per-node query indices,
                       concatenated; one entry per (node, query reading it)
  KVMapQ_List_Offset : (kv_num,) int32 — per-node start into KVMapQ_List
  KVMapQ_List_Len    : (kv_num,) int32 — per-node query count

The stage-1 kernel processes BLOCK_M=32 queries per (head, node) CTA, so
nodes with > 32 readers must be split across multiple kv_num entries
(repeating the KV index range). DeftBackend handles the split.
"""

from .tree_attention import tree_attention_fwd  # noqa: F401

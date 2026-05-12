"""AttentionContext for the SHARED_*L_DEC_TAIL strategies.

Per-step state runs prefix(-and-optional-intermediate) attention via the
prefill kernel and the per-beam tail attention via the paged-decode
kernel, then merges via online softmax. The decode kernel is purpose-built
for CTA_Q=1, so the per-beam level has 0% padding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import flashinfer.page
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    merge_state_in_place,
)

from ...models.attention import AttentionContext
from ...page_table import PageTable

# Module-level: when BS_KERNEL_TRACE_DEC_TAIL_KERNELS=1, attend() times
# each sub-kernel (KV append, prefix, tail decode, merge) via CUDA
# events and appends ms-readings to this list. Read by benchmarks.
DEC_TAIL_KERNEL_TRACE: list[dict] = []


@dataclass
class DecodeTailCascadeContext(AttentionContext):
    """Hybrid prefill-cascade-prefix + decode-tail context.

    ``prefix_wrapper`` runs B groups of K queries against the shared
    prefix KV (level 0). ``inter_wrappers`` runs the optional
    intermediate levels (one wrapper per intermediate level, length
    ``depth - 2``). ``decode_wrapper`` runs B*K beams × 1 query against
    per-beam tail KV. All ``depth`` outputs are merged via
    ``merge_state_in_place``.

    K/V append goes to the per-beam tail page (one per beam, in the
    same cascade order as the wrappers' plans).
    """
    page_table: PageTable
    write_pi: torch.Tensor   # [B*K] int32 — per-beam tail page idx for K/V write
    write_po: torch.Tensor   # [B*K] int32 — offset within tail page
    prefix_wrapper: BatchPrefillWithPagedKVCacheWrapper
    decode_wrapper: BatchDecodeWithPagedKVCacheWrapper
    inter_wrappers: list = field(default_factory=list)
    _write_helper_indptr: torch.Tensor | None = field(default=None, repr=False)

    def attend(self, q, k, v, layer_idx):
        num_kv_heads = self.page_table.head_num
        head_dim = self.page_table.head_dim
        num_heads = q.shape[-1] // head_dim
        kv_cache = self.page_table.kv_cache_at_layer[layer_idx]
        kv_tuple = (kv_cache[0], kv_cache[1])

        k_3d = k.view(-1, num_kv_heads, head_dim)
        v_3d = v.view(-1, num_kv_heads, head_dim)
        nnz = k_3d.shape[0]
        device = k_3d.device
        if (
            self._write_helper_indptr is None
            or self._write_helper_indptr.shape[0] < nnz + 1
        ):
            self._write_helper_indptr = torch.arange(
                nnz + 1, dtype=torch.int32, device=device,
            )
        batch_idx = self._write_helper_indptr[:nnz]
        kv_indptr = self._write_helper_indptr[: nnz + 1]

        trace = bool(int(os.environ.get("BS_KERNEL_TRACE_DEC_TAIL_KERNELS", "0")))
        if trace:
            ev = lambda: torch.cuda.Event(enable_timing=True)
            e0, e_append, e_tail, e_prefix, e_merge = ev(), ev(), ev(), ev(), ev()
            e0.record()

        flashinfer.page.append_paged_kv_cache(
            append_key=k_3d,
            append_value=v_3d,
            batch_indices=batch_idx,
            positions=self.write_po,
            paged_kv_cache=kv_tuple,
            kv_indices=self.write_pi,
            kv_indptr=kv_indptr,
            kv_last_page_len=self.write_po,
            kv_layout="NHD",
        )
        if trace:
            e_append.record()

        q_3d = q.view(-1, num_heads, head_dim)

        # Run all sub-attentions with return_lse=True. Tail's output is
        # the in-place merge accumulator; prefix and intermediate are
        # folded in.
        out_tail, lse_tail = self.decode_wrapper.run(
            q_3d, kv_tuple, return_lse=True,
        )
        if trace:
            e_tail.record()

        out_pre, lse_pre = self.prefix_wrapper.run(
            q_3d, kv_tuple, return_lse=True,
        )
        if trace:
            e_prefix.record()

        merge_state_in_place(out_tail, lse_tail, out_pre, lse_pre)
        for inter_w in self.inter_wrappers:
            out_int, lse_int = inter_w.run(
                q_3d, kv_tuple, return_lse=True,
            )
            merge_state_in_place(out_tail, lse_tail, out_int, lse_int)
        if trace:
            e_merge.record()
            torch.cuda.synchronize()
            DEC_TAIL_KERNEL_TRACE.append({
                "layer_idx": int(layer_idx),
                "append_ms": e0.elapsed_time(e_append),
                "tail_ms":   e_append.elapsed_time(e_tail),
                "prefix_ms": e_tail.elapsed_time(e_prefix),
                "merge_ms":  e_prefix.elapsed_time(e_merge),
            })

        return out_tail.reshape(*q.shape[:-1], num_heads * head_dim)

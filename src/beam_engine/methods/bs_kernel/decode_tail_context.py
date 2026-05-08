"""AttentionContext for the SHARED_*L_DEC_TAIL strategies.

Per-step state runs prefix(-and-optional-intermediate) attention via the
prefill kernel and the per-beam tail attention via the paged-decode
kernel, then merges via online softmax. The decode kernel is purpose-built
for CTA_Q=1, so the per-beam level has 0% padding.
"""

from __future__ import annotations

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


@dataclass
class DecodeTailCascadeContext(AttentionContext):
    """Hybrid prefill-cascade-prefix + decode-tail context.

    ``prefix_wrapper`` runs B groups of K queries against the shared
    prefix KV. ``inter_wrapper`` (optional, 3L only) runs G groups
    against intermediate KV. ``decode_wrapper`` runs B*K beams × 1
    query against per-beam tail KV. All three (or two, at 2L) outputs
    are merged via ``merge_state_in_place``.

    K/V append goes to the per-beam tail page (one per beam, in the
    same cascade order as the wrappers' plans).
    """
    page_table: PageTable
    write_pi: torch.Tensor   # [B*K] int32 — per-beam tail page idx for K/V write
    write_po: torch.Tensor   # [B*K] int32 — offset within tail page
    prefix_wrapper: BatchPrefillWithPagedKVCacheWrapper
    decode_wrapper: BatchDecodeWithPagedKVCacheWrapper
    inter_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = None
    _write_helper_indptr: torch.Tensor | None = field(default=None, repr=False)

    def attend(self, q, k, v, layer_idx):
        num_kv_heads = self.page_table.head_num
        head_dim = self.page_table.head_dim
        num_heads = q.shape[-1] // head_dim
        kv_cache = self.page_table.kv_cache_at_layer[layer_idx]

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
        flashinfer.page.append_paged_kv_cache(
            append_key=k_3d,
            append_value=v_3d,
            batch_indices=batch_idx,
            positions=self.write_po,
            paged_kv_cache=kv_cache,
            kv_indices=self.write_pi,
            kv_indptr=kv_indptr,
            kv_last_page_len=self.write_po,
            kv_layout="NHD",
        )

        q_3d = q.view(-1, num_heads, head_dim)

        # Run all sub-attentions with return_lse=True. Tail's output is
        # the in-place merge accumulator; prefix and intermediate are
        # folded in.
        out_tail, lse_tail = self.decode_wrapper.run(
            q_3d, kv_cache, return_lse=True,
        )
        out_pre, lse_pre = self.prefix_wrapper.run(
            q_3d, kv_cache, return_lse=True,
        )
        merge_state_in_place(out_tail, lse_tail, out_pre, lse_pre)
        if self.inter_wrapper is not None:
            out_int, lse_int = self.inter_wrapper.run(
                q_3d, kv_cache, return_lse=True,
            )
            merge_state_in_place(out_tail, lse_tail, out_int, lse_int)

        return out_tail.reshape(*q.shape[:-1], num_heads * head_dim)

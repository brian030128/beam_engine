"""Paged KV cache.

Pages are physical fixed-size blocks (``page_size`` tokens each) holding KV
entries for one transformer layer. ``allocate_block()`` and ``free_block()``
manage the free list. ``copy_block()`` clones a populated page (used for
copy-on-write at beam divergence).

Layout per layer: ``[2, max_num_pages, page_size, num_kv_heads, head_dim]``
where ``kv[0]`` is the K cache and ``kv[1]`` is the V cache. Both are
contiguous tensors. This is equivalent to FlashInfer's NHD format split
into two top-level slabs — wrappers consume them as a ``(k_cache, v_cache)``
tuple. The split form lets baselines that need a flat slot-indexed K/V
view (FastTree, DeFT) reshape ``kv[0]`` to ``[max_num_pages * page_size,
num_kv_heads, head_dim]`` without copying.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .logger import get_logger

logger = get_logger(__name__)


class PageTable:
    def __init__(
        self,
        layer_num: int,
        page_size: int = 16,
        max_num_pages: int = 1024,
        head_num: int = 32,
        head_dim: int = 128,
        v_head_dim: Optional[int] = None,
        device: torch.device | None = None,
        store_dtype: torch.dtype = torch.float16,
    ):
        self.layer_num = layer_num
        self.page_size = page_size
        self.max_num_pages = max_num_pages
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim or head_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.store_dtype = store_dtype

        self.free_pages: list[int] = list(range(max_num_pages))
        self.allocated_pages: set[int] = set()

        # One contiguous CUDA allocation for all layers, sliced into
        # per-layer views. This avoids ``layer_num`` separate
        # multi-GB-class allocations interleaved with model weights and
        # activations — the prior pattern fragmented the allocator enough
        # to push 8B / B=32 / K=32 / L_p=4 K / max_new=256 cells over the
        # 80 GB H100 budget even when the analytical KV+model+activation
        # total fit comfortably.
        #
        # Per-layer ``kv_cache_at_layer[i]`` is ``unified[i]`` — a
        # zero-copy contiguous view of shape ``[2, max_pages, page_size,
        # head_num, head_dim]``, identical to what callers used to
        # receive when each layer had its own tensor. The flat
        # slot-indexed view used by FastTree/DeFT (``kv[0].view(...)``)
        # also still works because ``kv[0]`` is itself contiguous within
        # the parent.
        self._unified_kv = torch.zeros(
            (layer_num, 2, max_num_pages, page_size, head_num, head_dim),
            dtype=store_dtype,
            device=self.device,
        )
        self.kv_cache_at_layer: List[torch.Tensor] = [
            self._unified_kv[i] for i in range(layer_num)
        ]

    def reset(self) -> None:
        self.free_pages = list(range(self.max_num_pages))
        self.allocated_pages = set()

    def allocate_block(self) -> int:
        if not self.free_pages:
            raise MemoryError("No free pages available")
        # ``pop()`` is O(1); ``pop(0)`` was O(N) and made slot-boundary
        # decode steps quadratic in max_num_pages. The free list is a
        # stack (LIFO); page-id ordering doesn't matter for correctness.
        page_idx = self.free_pages.pop()
        self.allocated_pages.add(page_idx)
        return page_idx

    def allocate_blocks(self, n: int) -> list[int]:
        """Pop ``n`` free page IDs in one shot.

        Equivalent to calling ``allocate_block()`` ``n`` times but
        avoids ``n`` Python method calls and ``n`` ``set.add`` calls
        on the hot slot-boundary path. Each decode step at
        ``current_pos % page_size == 0`` allocates B*K new pages —
        with B=32, K=64 that's 2048 allocations per affected step.
        """
        if len(self.free_pages) < n:
            raise MemoryError(
                f"Need {n} free pages, only {len(self.free_pages)} available"
            )
        out = self.free_pages[-n:]
        del self.free_pages[-n:]
        self.allocated_pages.update(out)
        return out

    def copy_block(self, page_idx: int, length: int) -> int:
        if length > self.page_size:
            raise ValueError(f"length ({length}) > page_size ({self.page_size})")
        if page_idx >= self.max_num_pages:
            raise ValueError(f"page idx ({page_idx}) out of bound ({self.max_num_pages})")
        new_page = self.allocate_block()
        for layer in range(self.layer_num):
            kv = self.kv_cache_at_layer[layer]
            # kv shape: [2, max_pages, page_size, ...].
            kv[:, new_page, :length].copy_(kv[:, page_idx, :length])
        return new_page

    def free_block(self, page_idx: int) -> None:
        if page_idx not in self.allocated_pages:
            logger.warning(f"Page {page_idx} is not allocated")
            return
        self.allocated_pages.remove(page_idx)
        self.free_pages.append(page_idx)

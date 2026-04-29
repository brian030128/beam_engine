"""Paged KV cache.

Pages are physical fixed-size blocks (``page_size`` tokens each) holding KV
entries for one transformer layer. ``allocate_block()`` and ``free_block()``
manage the free list. ``copy_block()`` clones a populated page (used for
copy-on-write at beam divergence). The cache uses FlashInfer's 5D NHD layout:
``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` per layer, where
index 0 holds keys and index 1 holds values.
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

        self.kv_cache_at_layer: List[torch.Tensor] = [
            torch.zeros(
                (max_num_pages, 2, page_size, head_num, head_dim),
                dtype=store_dtype,
                device=self.device,
            )
            for _ in range(layer_num)
        ]

    def reset(self) -> None:
        self.free_pages = list(range(self.max_num_pages))
        self.allocated_pages = set()

    def allocate_block(self) -> int:
        if not self.free_pages:
            raise MemoryError("No free pages available")
        page_idx = self.free_pages.pop(0)
        self.allocated_pages.add(page_idx)
        return page_idx

    def copy_block(self, page_idx: int, length: int) -> int:
        if length > self.page_size:
            raise ValueError(f"length ({length}) > page_size ({self.page_size})")
        if page_idx >= self.max_num_pages:
            raise ValueError(f"page idx ({page_idx}) out of bound ({self.max_num_pages})")
        new_page = self.allocate_block()
        for layer in range(self.layer_num):
            kv = self.kv_cache_at_layer[layer]
            kv[new_page, :, :length].copy_(kv[page_idx, :, :length])
        return new_page

    def free_block(self, page_idx: int) -> None:
        if page_idx not in self.allocated_pages:
            logger.warning(f"Page {page_idx} is not allocated")
            return
        self.allocated_pages.remove(page_idx)
        self.free_pages.append(page_idx)

"""
Simple Page Table for KV Cache Management

A minimal page table implementation with three core functions:
- allocate_block(): allocate a page and return page index
- free_block(page_idx): free a page by index
- write_block(layer, page_idx, tokens, index): write tokens to a specific page and layer
"""

import torch
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class PageTable:
    """
    Simple page table for managing KV cache pages across multiple layers.

    Each page can store page_size tokens, and pages can be allocated/freed independently.
    """

    def __init__(
        self,
        layer_num: int,
        page_size: int = 64,
        max_num_pages: int = 1024,
        head_num: int = 32,
        head_dim: int = 128,
        v_head_dim: Optional[int] = None,
        device: torch.device = None,
        store_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the page table.

        Args:
            layer_num: Number of transformer layers
            page_size: Number of tokens per page
            max_num_pages: Maximum number of pages available
            head_num: Number of attention heads
            head_dim: Dimension per attention head (for keys)
            v_head_dim: Dimension per attention head for values (defaults to head_dim)
            device: Device to allocate tensors on
            store_dtype: Data type for KV cache tensors
        """
        self.layer_num = layer_num
        self.page_size = page_size
        self.max_num_pages = max_num_pages
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim or head_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.store_dtype = store_dtype

        # Simple free page tracking
        self.free_pages = list(range(max_num_pages))
        self.allocated_pages = set()

        # Physical page storage - per layer using FlashInfer 5D format
        # [total_num_pages, 2, page_size, num_kv_heads, head_dim]
        # Index 0 = keys, Index 1 = values
        self.kv_cache_at_layer: List[torch.Tensor] = [
            torch.zeros(
                (max_num_pages, 2, page_size, head_num, head_dim),
                dtype=store_dtype,
                device=self.device,
            )
            for _ in range(layer_num)
        ]

        logger.info(f"Initialized PageTable: {layer_num} layers, {max_num_pages} pages of size {page_size}")

    def allocate_block(self) -> int:
        """
        Allocate a page block and return its index.

        Returns:
            page_idx: Index of the allocated page

        Raises:
            MemoryError: If no free pages are available
        """
        if not self.free_pages:
            raise MemoryError("No free pages available")

        page_idx = self.free_pages.pop(0)
        self.allocated_pages.add(page_idx)

        logger.debug(f"Allocated page {page_idx}")
        return page_idx

    def free_block(self, page_idx: int):
        """
        Free a page block by its index.

        Args:
            page_idx: Index of the page to free
        """
        if page_idx not in self.allocated_pages:
            logger.warning(f"Page {page_idx} is not allocated")
            return

        self.allocated_pages.remove(page_idx)
        self.free_pages.append(page_idx)

        logger.debug(f"Freed page {page_idx}")

    def write_block(
        self,
        layer: int,
        page_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        index: int
    ):
        """
        Write key-value tokens to a specific page and layer.

        Args:
            layer: Layer index (0 to layer_num-1)
            page_idx: Page index to write to
            key: Key tensor [seq_len, head_num, head_dim]
            value: Value tensor [seq_len, head_num, v_head_dim]
            index: Number index to start writing tokens to (usually last page length)
        """
        assert key.shape == value.shape
        assert key.shape[0] + index <= self.page_size, f"Cannot fit {key.shape[0]} tokens at index {index} in page size {self.page_size}"

        if layer >= self.layer_num:
            raise ValueError(f"Layer {layer} out of range (max: {self.layer_num-1})")

        if page_idx not in self.allocated_pages:
            raise ValueError(f"Page {page_idx} is not allocated")

        if index > self.page_size:
            raise ValueError(f"Index {index} exceeds page size {self.page_size}")

        seq_len = key.shape[0]  # Number of tokens to write

        if seq_len > 0:
            # Write to the specific layer and page using 5D tensor format
            # kv_cache_at_layer[layer]: [total_num_pages, 2, page_size, num_kv_heads, head_dim]
            # key/value should be [seq_len, num_kv_heads, head_dim]
            self.kv_cache_at_layer[layer][page_idx, 0, index:index + seq_len, :, :] = key
            self.kv_cache_at_layer[layer][page_idx, 1, index:index + seq_len, :, :] = value

        logger.debug(f"Wrote {seq_len} tokens to layer {layer}, page {page_idx}")
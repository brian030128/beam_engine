"""FastTree attention kernel for efficient beam search decode."""

from .attn_kernels import fasttree_decode
from .metadata import FastTreeMetadata

__all__ = ["fasttree_decode", "FastTreeMetadata"]

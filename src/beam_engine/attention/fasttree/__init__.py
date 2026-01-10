"""
FastTree attention kernel for tree-structured decoding.

Vendored from FastTree-Artifact SGLang plugin.
"""

from .metadata import FastTreeMetadata
from .kernel import fasttree_decode
from .preparation import prepare_fasttree_metadata_for_paged_cache

__all__ = [
    "FastTreeMetadata",
    "fasttree_decode",
    "prepare_fasttree_metadata_for_paged_cache",
]

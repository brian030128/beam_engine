"""
Attention mode enumeration for FlashInfer kernel selection.
"""

from enum import Enum

class AttentionMode(Enum):
    """
    Enum for different FlashInfer attention computation modes.

    PREFILL: Used for initial prompt processing with batch prefill kernel
    DECODE: Used for token-by-token generation with batch decode kernel
    """
    PREFILL = "prefill"
    DECODE = "decode"
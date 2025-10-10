"""
Embedding Provider Types
========================
Enumerates available embedding provider backends.
"""

from enum import Enum, auto


class ProviderType(Enum):
    """
    Enumeration of supported provider backends.
    Single Responsibility: identify provider selection options.
    """

    GEMMA_LOCAL = auto()
    BGE3_LOCAL = auto()
    GEMMA_API = auto()
    BGE3_API = auto()

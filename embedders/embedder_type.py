"""
Embedder Type Enumeration
=========================
Categorizes embedder families for factory creation.
"""

from enum import Enum, auto


class EmbedderType(Enum):
    """
    Enumeration to represent embedder families.
    Single Responsibility: provide factory-friendly identifiers.
    """

    GEMMA = auto()
    BGE3 = auto()
    ROUTER = auto()
    OLLAMA = auto()

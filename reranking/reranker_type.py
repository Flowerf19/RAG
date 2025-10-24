"""
Reranker Type Enumeration
==========================
Định nghĩa các loại reranker có sẵn
"""

from enum import Enum


class RerankerType(Enum):
    """Enum for available reranker types"""
    BGE_RERANKER = "bge_reranker"  # Local BGE reranker
    COHERE = "cohere"  # Cohere API reranker
    JINA = "jina"  # Jina API reranker

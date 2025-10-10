"""
Embedding Models Package
========================
Exports normalized data contracts used by embedding providers.
"""

from .embed_request import EmbedRequest
from .embedding_result import EmbeddingResult

__all__ = [
    'EmbedRequest',
    'EmbeddingResult',
]

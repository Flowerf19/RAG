"""
Embedding Providers Package
===========================
Exports concrete embedding providers.
"""

from .bge3_embedder import BGE3Embedder
from .gemma_embedder import GemmaEmbedder
from .ollama_embedder import OllamaEmbedder

__all__ = [
    'BGE3Embedder',
    'GemmaEmbedder',
    'OllamaEmbedder',
]

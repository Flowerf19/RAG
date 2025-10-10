"""
Embedders Package
=================
Provider-agnostic embedding orchestration.
"""

from .chunkset_embedder import ChunkSetEmbedder
from .embedder_factory import EmbedderFactory
from .embedder_type import EmbedderType
from .embedding_profile import EmbeddingProfile
from .i_embedder import IEmbedder
from .model.embed_request import EmbedRequest
from .model.embedding_result import EmbeddingResult
from .provider_router import ProviderRouter
from .provider_type import ProviderType
from .providers import BGE3Embedder, GemmaEmbedder, OllamaEmbedder

__all__ = [
    'ChunkSetEmbedder',
    'EmbedderFactory',
    'EmbedderType',
    'EmbeddingProfile',
    'IEmbedder',
    'EmbedRequest',
    'EmbeddingResult',
    'ProviderRouter',
    'ProviderType',
    'BGE3Embedder',
    'GemmaEmbedder',
    'OllamaEmbedder',
]

"""
Embedder Manager for Backend Dashboard
Provides cached embedder creation for evaluation.
"""

from typing import Dict, Any
from embedders.embedder_factory import EmbedderFactory
from embedders.embedder_type import EmbedderType
from embedders.model.embedding_profile import EmbeddingProfile


def get_or_create_embedder(cache: Dict[str, Any], embedder_type: str):
    """
    Get cached embedder or create new one.

    Args:
        cache: Cache dictionary to store embedders
        embedder_type: Type of embedder ('ollama', 'huggingface_local', etc.)

    Returns:
        Embedder instance
    """
    if embedder_type in cache:
        return cache[embedder_type]

    factory = EmbedderFactory()

    if embedder_type == 'ollama':
        # Use BGE-M3 Ollama embedder (same as pipeline)
        profile = EmbeddingProfile(
            model_id='bge-m3:latest',
            provider='ollama',
            max_tokens=8192,
            dimension=1024
        )
        embedder = factory.create(EmbedderType.OLLAMA, profile)
    elif embedder_type == 'huggingface_local':
        # Use BGE-M3 local
        profile = EmbeddingProfile(
            model_id='BAAI/bge-m3',
            provider='huggingface',
            max_tokens=8192,
            dimension=1024
        )
        embedder = factory.create(EmbedderType.HUGGINGFACE, profile, use_api=False)
    else:
        # Default to Ollama BGE-M3
        profile = EmbeddingProfile(
            model_id='bge-m3:latest',
            provider='ollama',
            max_tokens=8192,
            dimension=1024
        )
        embedder = factory.create(EmbedderType.OLLAMA, profile)

    # Cache the embedder
    cache[embedder_type] = embedder
    return embedder
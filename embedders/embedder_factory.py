"""
Embedding Factory
=================
Creates embedder instances from configuration.
"""

from typing import Dict

from .embedder_type import EmbedderType
from .embedding_profile import EmbeddingProfile
from .i_embedder import IEmbedder
from .provider_router import ProviderRouter
from .provider_type import ProviderType
from .providers import BGE3Embedder, GemmaEmbedder, OllamaEmbedder


class EmbedderFactory:
    """
    Factory to build embedders based on profile and type.
    Single Responsibility: centralize embedder instantiation logic.
    """

    def __init__(self, registry: Dict[ProviderType, IEmbedder] | None = None):
        self._registry = registry or {}

    def create(self, embedder_type: EmbedderType, profile: EmbeddingProfile) -> IEmbedder:
        """Create the requested embedder."""
        if embedder_type == EmbedderType.GEMMA:
            return GemmaEmbedder(
                model_name=profile.model_id,
                max_tokens=profile.max_tokens,
                normalize=profile.normalize,
            )
        if embedder_type == EmbedderType.BGE3:
            return BGE3Embedder(
                model_name=profile.model_id,
                max_tokens=profile.max_tokens,
                normalize=profile.normalize,
            )
        if embedder_type == EmbedderType.ROUTER:
            router = ProviderRouter(active=None, registry=self._registry.copy())
            router.configure(profile)
            return router
        if embedder_type == EmbedderType.OLLAMA:
            return OllamaEmbedder(
                model_name=profile.model_id,
                max_tokens=profile.max_tokens,
                normalize=profile.normalize,
                endpoint=profile.endpoint,
                timeout=profile.timeout,
            )
        raise ValueError(f"Unsupported embedder type: {embedder_type!r}")

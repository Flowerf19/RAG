"""
Provider Router
===============
Routes embedding requests to the active provider or selects one dynamically.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .embedding_profile import EmbeddingProfile
from .i_embedder import IEmbedder
from .model.embed_request import EmbedRequest
from .model.embedding_result import EmbeddingResult
from .provider_type import ProviderType


class ProviderRouter(IEmbedder):
    """
    Router that forwards calls to the selected provider.
    Single Responsibility: manage provider switching and routing policies.
    """

    def __init__(self, active: Optional[ProviderType], registry: Dict[ProviderType, IEmbedder]):
        if not registry:
            raise ValueError("Provider registry must not be empty.")
        self._registry = registry
        self._active = active or next(iter(registry.keys()))

    @property
    def model_name(self) -> str:
        return self._registry[self._active].model_name

    @property
    def device(self) -> str:
        return self._registry[self._active].device

    @property
    def max_tokens(self) -> int:
        return self._registry[self._active].max_tokens

    def configure(self, profile: EmbeddingProfile):
        """
        Configure router based on profile preferences.
        Picks first available provider respecting preferences.
        """
        if not profile.provider_prefs:
            return
        for provider in profile.provider_prefs:
            if provider in self._registry:
                self._active = provider
                return
        raise ValueError("No preferred providers are available in the registry.")

    def switch_provider(self, provider: ProviderType):
        """Switch active provider."""
        if provider not in self._registry:
            raise ValueError(f"Provider {provider} not registered.")
        self._active = provider

    def available(self) -> List[ProviderType]:
        """Return available provider types."""
        return list(self._registry.keys())

    def route(self, req: EmbedRequest) -> ProviderType:
        """
        Hook for advanced routing decisions.
        Currently picks active provider, can be extended for language routing.
        """
        return self._active

    def _resolve(self, req: Optional[EmbedRequest] = None) -> IEmbedder:
        """Return the provider chosen for the given request."""
        provider = self.route(req) if req else self._active
        return self._registry[provider]

    def get_dimensions(self) -> int:
        return self._resolve().get_dimensions()

    def embed(self, text: str):
        return self._resolve().embed(text)

    def embed_batch(self, texts: List[str]):
        return self._resolve().embed_batch(texts)

    def embed_request(self, req: EmbedRequest) -> EmbeddingResult:
        provider = self._resolve(req)
        return provider.embed_request(req)

    def embed_batch_req(self, reqs: List[EmbedRequest]) -> List[EmbeddingResult]:
        if not reqs:
            return []
        # Simple grouping by routed provider to minimize switches.
        results: List[EmbeddingResult] = []
        grouped: Dict[ProviderType, List[EmbedRequest]] = {}
        for req in reqs:
            provider_type = self.route(req)
            grouped.setdefault(provider_type, []).append(req)
        for provider_type, batch in grouped.items():
            provider = self._registry[provider_type]
            results.extend(provider.embed_batch_req(batch))
        return results

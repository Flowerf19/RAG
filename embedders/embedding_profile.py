"""
Embedding Profile
=================
Specifies model configuration for embedder instantiation.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .provider_type import ProviderType


@dataclass
class EmbeddingProfile:
    """
    Holds desired characteristics for an embedder instance.
    Single Responsibility: capture configuration to create embedders.
    """

    model_id: str
    dimension: int
    max_tokens: int
    pooling: str = "mean"
    normalize: bool = True
    provider_prefs: List[ProviderType] = field(default_factory=list)
    endpoint: Optional[str] = None
    timeout: float = 60.0
    batch_size: int = 8

    def prefers(self, provider: ProviderType) -> bool:
        """Check whether a provider is part of the preference list."""
        return not self.provider_prefs or provider in self.provider_prefs

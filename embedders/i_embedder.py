"""
Embedding Interface Definition
==============================
Defines the contract for all embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List

from .model.embed_request import EmbedRequest
from .model.embedding_result import EmbeddingResult


class IEmbedder(ABC):
    """
    Interface that all embedding providers must implement.
    Single Responsibility: define operations for generating embeddings.
    """

    model_name: str
    device: str
    max_tokens: int

    @abstractmethod
    def get_dimensions(self) -> int:
        """Return the dimension of the embedding vector."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text snippet."""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text snippets."""

    @abstractmethod
    def embed_request(self, req: EmbedRequest) -> EmbeddingResult:
        """Embed using an EmbedRequest structure."""

    @abstractmethod
    def embed_batch_req(self, reqs: List[EmbedRequest]) -> List[EmbeddingResult]:
        """Embed a batch of EmbedRequest objects."""

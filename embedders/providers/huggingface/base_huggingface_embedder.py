"""
Base HuggingFace Embedding Provider
===================================
Base class cho tất cả HuggingFace embedding providers.
Tương tự BaseOllamaEmbedder nhưng cho HuggingFace.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..base_embedder import BaseEmbedder
from ...model.embedding_profile import EmbeddingProfile

logger = logging.getLogger(__name__)


class BaseHuggingFaceEmbedder(BaseEmbedder, ABC):
    """
    Base class cho tất cả HuggingFace embedding providers.
    Single Responsibility: Cung cấp common functionality cho HF embedders.
    
    Subclasses:
    - HuggingFaceApiEmbedder: sử dụng Inference API
    - HuggingFaceLocalEmbedder: sử dụng transformers local
    """

    # Class-level constants (được override bởi subclass)
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DIMENSION = 384
    MAX_TOKENS = 512
    PROVIDER = "huggingface"

    def __init__(self,
                 profile: Optional[EmbeddingProfile] = None,
                 model_name: Optional[str] = None):
        """
        Initialize base HuggingFace embedder.

        Args:
            profile: Embedding profile, nếu None sẽ tạo từ class constants
            model_name: Override model name from class constant
        """
        if profile is None:
            profile = self._create_profile(model_name)

        super().__init__(profile)
        self.model_name = model_name or self.MODEL_NAME

    @classmethod
    def _create_profile(cls, model_name: Optional[str] = None) -> EmbeddingProfile:
        """
        Tạo EmbeddingProfile từ class constants.
        Config nằm TRONG class, không phụ thuộc external config.
        
        Args:
            model_name: Optional model name override
            
        Returns:
            EmbeddingProfile: Configured profile
        """
        final_model_name = model_name or cls.MODEL_NAME
        model_id = f"hf_{final_model_name.replace('/', '_').replace('-', '_')}"
        
        return EmbeddingProfile(
            model_id=model_id,
            provider=cls.PROVIDER,
            max_tokens=cls.MAX_TOKENS,
            dimension=cls.DIMENSION,
            normalize=True
        )

    @property
    def dimension(self) -> int:
        """
        Lấy dimension của HuggingFace embedding.

        Returns:
            int: Embedding dimension
        """
        return self.DIMENSION

    @abstractmethod
    def _generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding using HuggingFace (API or local).
        Must be implemented by subclass.

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector
        """
        pass

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            List[float]: Embedding vector
        """
        return self._generate_embedding(text)

    @classmethod
    @abstractmethod
    def create_default(cls, **kwargs) -> 'BaseHuggingFaceEmbedder':
        """
        Factory method để tạo embedder với cấu hình mặc định.
        Must be implemented by subclass.
        
        Returns:
            BaseHuggingFaceEmbedder: Configured embedder
        """
        pass

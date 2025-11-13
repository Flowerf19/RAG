"""
Embedder Interface
==================
Abstract interface cho tất cả embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List


class IEmbedder(ABC):
    """
    Interface cho embedding providers.
    Single Responsibility: Define contract cho embedding operations.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Lấy dimension của embedding vectors.

        Returns:
            int: Embedding dimension
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Tạo embedding cho một text string.

        Args:
            text: Input text để embed

        Returns:
            List[float]: Embedding vector
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Lấy tên cụ thể của model đang được sử dụng.

        Returns:
            str: Model name (e.g., "gemma-7b", "bge-m3", "e5-large-instruct")
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test connection tới embedding service.

        Returns:
            bool: True nếu connection thành công
        """
        pass
"""
Base Reranker Interface
=======================
Abstract interface cho tất cả rerankers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class RerankResult:
    """Result từ reranking"""
    index: int  # Index trong list gốc
    score: float  # Rerank score
    document: str  # Text content
    metadata: Dict[str, Any] = None  # Original metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RerankerProfile:
    """Thông tin về reranker model"""
    model_id: str
    provider: str  # "bge", "cohere", "jina"
    max_query_length: int = 512
    max_document_length: int = 512
    is_local: bool = True


class IReranker(ABC):
    """Interface for all reranker implementations"""
    
    @property
    @abstractmethod
    def profile(self) -> RerankerProfile:
        """Get reranker profile information"""
        pass
    
    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[RerankResult]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Query text
            documents: List of document texts to rerank
            top_k: Number of top results to return
            
        Returns:
            List of RerankResult sorted by score (highest first)
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if reranker is accessible.
        
        Returns:
            True if connection successful
        """
        pass

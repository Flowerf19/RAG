"""
Reranker Factory
================
Factory for creating reranker instances
"""

import logging
from typing import Optional

from reranking.reranker_type import RerankerType
from reranking.i_reranker import IReranker

logger = logging.getLogger(__name__)


class RerankerFactory:
    """Factory for creating reranker instances"""
    
    @staticmethod
    def create_bge_local(
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu"
    ) -> IReranker:
        """
        Create BGE local reranker.
        
        Args:
            model_name: Model name
            device: Device to run on
            
        Returns:
            BGE reranker instance
        """
        from reranking.providers.bge_reranker_local import BGERerankerLocal
        return BGERerankerLocal(model_name=model_name, device=device)
    
    @staticmethod
    def create_cohere(
        api_token: str,
        model_name: str = "rerank-english-v3.0"
    ) -> IReranker:
        """
        Create Cohere API reranker.
        
        Args:
            api_token: Cohere API token
            model_name: Model name
            
        Returns:
            Cohere reranker instance
        """
        from reranking.providers.cohere_reranker import CohereReranker
        return CohereReranker(api_token=api_token, model_name=model_name)
    
    @staticmethod
    def create_jina(
        api_token: str,
        model_name: str = "jina-reranker-v2-base-multilingual"
    ) -> IReranker:
        """
        Create Jina API reranker.
        
        Args:
            api_token: Jina API token
            model_name: Model name
            
        Returns:
            Jina reranker instance
        """
        from reranking.providers.jina_reranker import JinaReranker
        return JinaReranker(api_token=api_token, model_name=model_name)
    
    @classmethod
    def create(
        cls,
        reranker_type: RerankerType,
        api_token: Optional[str] = None,
        model_name: Optional[str] = None,
        device: str = "cpu"
    ) -> IReranker:
        """
        Create reranker based on type.
        
        Args:
            reranker_type: Type of reranker
            api_token: API token (for API-based rerankers)
            model_name: Model name (optional, uses defaults)
            device: Device for local rerankers
            
        Returns:
            Reranker instance
        """
        if reranker_type == RerankerType.BGE_RERANKER:
            return cls.create_bge_local(model_name=model_name, device=device)
        
        elif reranker_type == RerankerType.COHERE:
            if not api_token:
                raise ValueError("API token required for Cohere reranker")
            return cls.create_cohere(api_token=api_token, model_name=model_name)
        
        elif reranker_type == RerankerType.JINA:
            if not api_token:
                raise ValueError("API token required for Jina reranker")
            return cls.create_jina(api_token=api_token, model_name=model_name)
        
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")

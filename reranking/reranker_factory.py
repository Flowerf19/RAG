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
        Create BGE local reranker (uses BGE3HuggingFaceLocalReranker).

        Args:
            model_name: Model name
            device: Device to run on

        Returns:
            BGE reranker instance
        """
        from reranking.providers.bge_m3_hf_local_reranker import BGE3HuggingFaceLocalReranker
        return BGE3HuggingFaceLocalReranker(model_name=model_name, device=device)
    
    @staticmethod
    def create_bge_m3_ollama(device: str = "auto") -> IReranker:
        """
        Create BGE-M3 HuggingFace reranker (no Ollama dependency).

        Args:
            device: Device to run on ('auto', 'cpu', 'cuda')

        Returns:
            BGE-M3 HF reranker instance
        """
        from reranking.providers.bge_m3_ollama_reranker import BGE3OllamaReranker
        return BGE3OllamaReranker(device=device)
    @staticmethod
    def create_bge_m3_hf_api(api_token: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> IReranker:
        """
        Create BGE-M3 HuggingFace API reranker.
        Uses sentence-transformers/all-MiniLM-L6-v2 (free, Apache-2.0, multilingual support).
        
        Args:
            api_token: HuggingFace API token
            model_name: Model name
            
        Returns:
            BGE-M3 HF API reranker instance
        """
        from reranking.providers.bge_m3_hf_api_reranker import BGE3HuggingFaceApiReranker
        return BGE3HuggingFaceApiReranker(api_token=api_token, model_name=model_name)
    
    @staticmethod
    def create_bge_m3_hf_local(model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu") -> IReranker:
        """
        Create BGE-M3 HuggingFace local reranker.
        
        Args:
            model_name: Model name
            device: Device to run on
            
        Returns:
            BGE-M3 HF local reranker instance
        """
        from reranking.providers.bge_m3_hf_local_reranker import BGE3HuggingFaceLocalReranker
        return BGE3HuggingFaceLocalReranker(model_name=model_name, device=device)
    
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
        
        elif reranker_type == RerankerType.BGE_M3_OLLAMA:
            return cls.create_bge_m3_ollama()
        
        elif reranker_type == RerankerType.BGE_M3_HF_API:
            if not api_token:
                raise ValueError("API token required for BGE-M3 HF API reranker")
            return cls.create_bge_m3_hf_api(api_token=api_token, model_name=model_name)
        
        elif reranker_type == RerankerType.BGE_M3_HF_LOCAL:
            return cls.create_bge_m3_hf_local(model_name=model_name, device=device)
        
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

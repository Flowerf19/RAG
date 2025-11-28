"""
Base API Reranker
=================
Base class cho API-based rerankers (Cohere, Jina, etc.)
"""

from typing import List, Optional
import logging
import requests
from reranking.i_reranker import IReranker, RerankResult, RerankerProfile

logger = logging.getLogger(__name__)


class BaseAPIReranker(IReranker):
    """Base class for API reranker implementations"""
    
    def __init__(self, api_token: str, model_name: str, api_base_url: str):
        """
        Initialize API reranker.
        
        Args:
            api_token: API authentication token
            model_name: Model name/identifier
            api_base_url: Base URL for API endpoint
        """
        self.api_token = api_token
        self.model_name = model_name
        self.api_base_url = api_base_url
        self._profile = None
        
        if not api_token:
            raise ValueError("API token is required")
        
        self._initialize_profile()
    
    def _initialize_profile(self):
        """Initialize reranker profile - to be implemented by subclasses"""
        raise NotImplementedError("Subclass must implement _initialize_profile")
    
    @property
    def profile(self) -> RerankerProfile:
        """Get reranker profile"""
        if self._profile is None:
            raise RuntimeError("Profile not initialized")
        return self._profile
    
    def get_model_name(self) -> str:
        """Get specific reranker model name"""
        return self.profile.model_id if self.profile else self.model_name
    
    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[RerankResult]:
        """
        Rerank documents using API.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of results to return
            
        Returns:
            List of RerankResult sorted by score
        """
        if not documents:
            return []
        
        try:
            # Call API - to be implemented by subclasses
            api_results = self._call_api(query, documents, top_k)
            
            # Convert to RerankResult format
            results = [
                RerankResult(
                    index=result.get("index", idx),
                    score=result.get("score", 0.0),
                    document=result.get("document", documents[result.get("index", idx)])
                )
                for idx, result in enumerate(api_results)
            ]
            
            return results
            
        except Exception as e:
            logger.error(f"API rerank error: {e}")
            # Return original order with zero scores
            return [
                RerankResult(index=idx, score=0.0, document=doc)
                for idx, doc in enumerate(documents[:top_k])
            ]
    
    def _call_api(self, query: str, documents: List[str], top_k: int) -> List[dict]:
        """Call API endpoint - to be implemented by subclasses"""
        raise NotImplementedError("Subclass must implement _call_api")
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            # Simple test with minimal data
            test_query = "test"
            test_docs = ["test document"]
            self._call_api(test_query, test_docs, top_k=1)
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

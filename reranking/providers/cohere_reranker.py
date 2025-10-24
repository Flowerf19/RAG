"""
Cohere Reranker API Implementation
===================================
Cohere rerank API: https://cohere.com/rerank
"""

from typing import List
import logging
import requests

from reranking.providers.base_api_reranker import BaseAPIReranker
from reranking.i_reranker import RerankerProfile

logger = logging.getLogger(__name__)


class CohereReranker(BaseAPIReranker):
    """
    Cohere Reranker API Implementation.
    Free tier: 1000 requests/month
    """
    
    API_BASE_URL = "https://api.cohere.ai/v1"
    DEFAULT_MODEL = "rerank-english-v3.0"
    MAX_LENGTH = 512
    
    def __init__(self, api_token: str, model_name: str = None):
        """
        Initialize Cohere reranker.
        
        Args:
            api_token: Cohere API token
            model_name: Model name (default: rerank-english-v3.0)
        """
        if model_name is None:
            model_name = self.DEFAULT_MODEL
        
        logger.info(f"🔄 Initializing Cohere reranker: {model_name}")
        super().__init__(api_token, model_name, self.API_BASE_URL)
        logger.info("✅ Cohere reranker initialized")
    
    def _initialize_profile(self):
        """Initialize Cohere reranker profile"""
        self._profile = RerankerProfile(
            model_id=self.model_name,
            provider="cohere",
            max_query_length=self.MAX_LENGTH,
            max_document_length=self.MAX_LENGTH,
            is_local=False
        )
    
    def _call_api(self, query: str, documents: List[str], top_k: int) -> List[dict]:
        """
        Call Cohere rerank API.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of results to return
            
        Returns:
            List of dicts with index, score, document
        """
        url = f"{self.api_base_url}/rerank"
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
            "return_documents": True
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse results
            results = []
            for item in data.get("results", []):
                results.append({
                    "index": item.get("index"),
                    "score": item.get("relevance_score", 0.0),
                    "document": item.get("document", {}).get("text", documents[item.get("index", 0)])
                })
            
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Cohere API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing Cohere response: {e}")
            raise

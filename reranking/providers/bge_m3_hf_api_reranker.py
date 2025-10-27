"""
BGE M3 HuggingFace API Reranker
================================
BGE-M3 reranker using HuggingFace API
"""

import logging
from typing import List
import requests

from reranking.providers.base_api_reranker import BaseAPIReranker
from reranking.i_reranker import RerankerProfile

logger = logging.getLogger(__name__)


class BGE3HuggingFaceApiReranker(BaseAPIReranker):
    """
    BGE-M3 reranker using HuggingFace API.
    Uses sentence-transformers/all-MiniLM-L6-v2 as HF API doesn't support BGE reranker models.
    This model is free, Apache-2.0 licensed, and returns direct similarity scores.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    API_BASE_URL = "https://api-inference.huggingface.co"

    def __init__(self, api_token: str, model_name: str = None):
        """
        Initialize BGE-M3 HF API reranker.

        Args:
            api_token: HuggingFace API token
            model_name: Model name (default: sentence-transformers/all-MiniLM-L6-v2)
        """
        if model_name is None:
            model_name = self.DEFAULT_MODEL

        super().__init__(api_token, model_name, self.API_BASE_URL)
        logger.info(f"🔄 Initializing BGE HF API reranker: {model_name}")

    def _initialize_profile(self):
        """Initialize reranker profile"""
        self._profile = RerankerProfile(
            model_id=self.model_name,
            provider="huggingface",
            max_query_length=512,
            max_document_length=512,
            is_local=False
        )

    def _call_api(self, query: str, documents: List[str], top_k: int) -> List[dict]:
        """
        Call HuggingFace API for reranking using sentence-transformers model.
        Uses sentence-transformers/all-MiniLM-L6-v2 which directly outputs similarity scores.
        """
        try:
            url = f"{self.api_base_url}/models/{self.model_name}"
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }

            # Sentence-transformers format: {"source_sentence": query, "sentences": [docs]}
            payload = {
                "inputs": {
                    "source_sentence": query,
                    "sentences": documents
                }
            }

            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                results = response.json()
                
                # Sentence-transformers returns direct similarity scores as a list
                normalized_results = []
                
                if isinstance(results, list) and len(results) == len(documents):
                    scores = [float(score) for score in results]
                    
                    # Create index-score pairs and sort by score
                    scored_docs = [(idx, score) for idx, score in enumerate(scores)]
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    
                    # Return top_k results
                    for idx, score in scored_docs[:top_k]:
                        normalized_results.append({
                            "index": idx,
                            "score": score,
                            "document": documents[idx]
                        })
                else:
                    logger.warning(f"Unexpected response format: {type(results)}, len={len(results) if isinstance(results, list) else 'N/A'}")
                    # Fallback: return documents with zero scores
                    normalized_results = [
                        {"index": idx, "score": 0.0, "document": doc}
                        for idx, doc in enumerate(documents[:top_k])
                    ]

                return normalized_results
            else:
                logger.warning(f"HF API reranking failed: {response.status_code} - {response.text[:200]}")
                return [
                    {"index": idx, "score": 0.0, "document": doc}
                    for idx, doc in enumerate(documents[:top_k])
                ]

        except Exception as e:
            logger.warning(f"HF API reranking error: {e}")
            return [
                {"index": idx, "score": 0.0, "document": doc}
                for idx, doc in enumerate(documents[:top_k])
            ]
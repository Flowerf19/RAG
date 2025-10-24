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
    Uses sentence-transformers reranking models.
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
    API_BASE_URL = "https://api-inference.huggingface.co"

    def __init__(self, api_token: str, model_name: str = None):
        """
        Initialize BGE-M3 HF API reranker.

        Args:
            api_token: HuggingFace API token
            model_name: Model name (default: BAAI/bge-reranker-v2-m3)
        """
        if model_name is None:
            model_name = self.DEFAULT_MODEL

        super().__init__(api_token, model_name, self.API_BASE_URL)
        logger.info(f"🔄 Initializing BGE-M3 HF API reranker: {model_name}")

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
        Call HuggingFace API for reranking.
        Note: This might not work if the model doesn't support reranking endpoint.
        """
        try:
            # Try the reranking endpoint first
            url = f"{self.api_base_url}/models/{self.model_name}"
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }

            payload = {
                "inputs": {
                    "query": query,
                    "documents": documents,
                    "top_k": min(top_k, len(documents))
                }
            }

            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                results = response.json()
                # Normalize results format
                normalized_results = []
                if isinstance(results, list):
                    for idx, result in enumerate(results):
                        if isinstance(result, dict):
                            normalized_results.append({
                                "index": result.get("index", idx),
                                "score": result.get("score", 0.0),
                                "document": result.get("document", documents[result.get("index", idx)])
                            })
                        else:
                            # Handle score-only format
                            normalized_results.append({
                                "index": idx,
                                "score": float(result) if isinstance(result, (int, float)) else 0.0,
                                "document": documents[idx]
                            })
                else:
                    # Fallback: return documents with zero scores
                    normalized_results = [
                        {"index": idx, "score": 0.0, "document": doc}
                        for idx, doc in enumerate(documents[:top_k])
                    ]

                return normalized_results
            else:
                # If reranking endpoint fails, return zero scores
                logger.warning(f"HF API reranking failed: {response.status_code} - {response.text}")
                return [
                    {"index": idx, "score": 0.0, "document": doc}
                    for idx, doc in enumerate(documents[:top_k])
                ]

        except Exception as e:
            logger.warning(f"HF API reranking error: {e}")
            # Return zero scores on error
            return [
                {"index": idx, "score": 0.0, "document": doc}
                for idx, doc in enumerate(documents[:top_k])
            ]
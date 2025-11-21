"""
BGE M3 HuggingFace API Reranker
================================
BGE-M3 reranker using HuggingFace API.

Uses sentence-transformers/all-MiniLM-L6-v2 as HF Inference API does not expose
the BGE reranking models directly. The model is free, Apache-2.0 licensed,
and returns cosine similarity scores.
"""

import logging
from typing import List, Optional

import requests

from reranking.providers.base_api_reranker import BaseAPIReranker
from reranking.i_reranker import RerankerProfile

logger = logging.getLogger(__name__)


class BGE3HuggingFaceApiReranker(BaseAPIReranker):
    """BGE-M3 reranker using HuggingFace Inference API."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    API_BASE_URL = "https://router.huggingface.co/hf-inference"

    def __init__(self, api_token: str, model_name: Optional[str] = None):
        if model_name is None:
            model_name = self.DEFAULT_MODEL

        super().__init__(api_token, model_name, self.API_BASE_URL)
        logger.info("Initializing BGE HF API reranker: %s", model_name)

    def _initialize_profile(self):
        """Initialize reranker profile."""
        self._profile = RerankerProfile(
            model_id=self.model_name,
            provider="huggingface",
            max_query_length=512,
            max_document_length=512,
            is_local=False,
        )

    def _call_api(self, query: str, documents: List[str], top_k: int) -> List[dict]:
        """
        Call HuggingFace API for reranking using a sentence-transformers model.

        The endpoint returns a list of similarity scores aligned to the provided documents.
        """
        try:
            url = f"{self.api_base_url}/models/{self.model_name}"
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            }

            payload = {"inputs": {"source_sentence": query, "sentences": documents}}

            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                results = response.json()
                normalized_results = []

                if isinstance(results, list) and len(results) == len(documents):
                    scores = [float(score) for score in results]
                    scored_docs = [(idx, score) for idx, score in enumerate(scores)]
                    scored_docs.sort(key=lambda x: x[1], reverse=True)

                    for idx, score in scored_docs[:top_k]:
                        normalized_results.append(
                            {"index": idx, "score": score, "document": documents[idx]}
                        )
                else:
                    logger.warning(
                        "Unexpected HF rerank response format: %s len=%s",
                        type(results),
                        len(results) if isinstance(results, list) else "N/A",
                    )
                    normalized_results = [
                        {"index": idx, "score": 0.0, "document": doc}
                        for idx, doc in enumerate(documents[:top_k])
                    ]

                return normalized_results

            logger.warning(
                "HF API reranking failed: %s - %s",
                response.status_code,
                response.text[:200],
            )
            return [
                {"index": idx, "score": 0.0, "document": doc}
                for idx, doc in enumerate(documents[:top_k])
            ]

        except Exception as exc:  # pragma: no cover - network/runtime failures
            logger.warning("HF API reranking error: %s", exc)
            return [
                {"index": idx, "score": 0.0, "document": doc}
                for idx, doc in enumerate(documents[:top_k])
            ]

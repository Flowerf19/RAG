"""
BGE M3 HuggingFace Local Reranker
==================================
BGE-M3 reranker using local HuggingFace transformers
"""

from typing import List
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reranking.providers.base_local_reranker import BaseLocalReranker
from reranking.i_reranker import RerankerProfile

logger = logging.getLogger(__name__)


class BGE3HuggingFaceLocalReranker(BaseLocalReranker):
    """
    BGE-M3 reranker using local HuggingFace transformers.
    Model: BAAI/bge-reranker-v2-m3
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
    MAX_LENGTH = 512

    def __init__(self, model_name: str = None, device: str = "cpu"):
        """
        Initialize BGE-M3 HF local reranker.

        Args:
            model_name: Model name (default: BAAI/bge-reranker-v2-m3)
            device: Device to run on
        """
        if model_name is None:
            model_name = self.DEFAULT_MODEL

        logger.info(f"🔄 Loading BGE-M3 HF local reranker: {model_name}")
        super().__init__(model_name, device)
        logger.info(f"✅ BGE-M3 HF local reranker loaded on {device}")

    def _load_model(self):
        """Load BGE-M3 reranker model and tokenizer"""
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.to(self.device)
            self._model.eval()

            self._profile = RerankerProfile(
                model_id=self.model_name,
                provider="huggingface",
                max_query_length=self.MAX_LENGTH,
                max_document_length=self.MAX_LENGTH,
                is_local=True
            )

        except Exception as e:
            logger.error(f"Failed to load BGE-M3 HF local reranker: {e}")
            raise

    def _compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute relevance scores using BGE-M3 reranker.

        Args:
            query: Query text
            documents: List of document texts

        Returns:
            List of relevance scores
        """
        try:
            # Prepare pairs: [(query, doc1), (query, doc2), ...]
            pairs = [[query, doc] for doc in documents]

            # Tokenize
            with torch.no_grad():
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_LENGTH,
                    return_tensors="pt"
                ).to(self.device)

                # Get scores
                scores = self._model(**inputs, return_dict=True).logits.view(-1).float()
                scores = scores.cpu().numpy().tolist()

            return scores

        except Exception as e:
            logger.error(f"Error computing scores: {e}")
            # Return zero scores on error
            return [0.0] * len(documents)
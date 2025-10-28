
"""
MS MARCO MiniLM-L6-v2 Local Reranker
====================================
Lightweight cross-encoder reranker that runs fully offline using a local
copy of the `cross-encoder/ms-marco-MiniLM-L-6-v2` model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reranking.providers.base_local_reranker import BaseLocalReranker
from reranking.i_reranker import RerankerProfile

logger = logging.getLogger(__name__)


class MSMARCOMiniLMLocalReranker(BaseLocalReranker):
    """
    Cross-encoder reranker based on ms-marco MiniLM-L-6-v2, loaded from disk.
    """

    DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DEFAULT_MODEL_DIR = (
        Path(__file__).resolve().parents[2] / "rerank_model" / "model"
    ).resolve()
    MAX_LENGTH = 512

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        *,
        device: str = "cpu",
    ) -> None:
        """
        Initialize MiniLM reranker.

        Args:
            model_path: Local directory containing the model weights. Falls back to
                        `rerank_model/model` if not provided.
            device: Torch device to run on ("cpu" or "cuda").
        """
        base_dir = Path(__file__).resolve().parents[2]
        if model_path:
            candidate = Path(model_path).expanduser()
            resolved_path = (
                (base_dir / candidate).resolve()
                if not candidate.is_absolute()
                else candidate.resolve()
            )
        else:
            resolved_path = self.DEFAULT_MODEL_DIR
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"MiniLM reranker model directory not found: {resolved_path}"
            )

        self._model_path = resolved_path

        logger.info("Loading MiniLM reranker from %s", resolved_path)
        super().__init__(str(resolved_path), device=device)
        logger.info("MiniLM reranker ready on %s", device)

    def _load_model(self) -> None:
        """Load MiniLM model and tokenizer from local storage."""
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.to(self.device)
            self._model.eval()

            self._profile = RerankerProfile(
                model_id=str(self._model_path),
                provider="local",
                max_query_length=self.MAX_LENGTH,
                max_document_length=self.MAX_LENGTH,
                is_local=True,
            )
        except Exception as exc:  # pragma: no cover - hardware/env specific
            logger.error("Failed to load MiniLM reranker: %s", exc)
            raise

    def _compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute scores for query-document pairs using the MiniLM cross encoder.
        """
        if not documents:
            return []

        try:
            pairs = [(query, doc) for doc in documents]
            with torch.no_grad():
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)

                logits = self._model(**inputs, return_dict=True).logits.view(-1)
                return logits.cpu().float().tolist()
        except Exception as exc:  # pragma: no cover - hardware/env specific
            logger.error("MiniLM reranker scoring error: %s", exc)
            return [0.0] * len(documents)

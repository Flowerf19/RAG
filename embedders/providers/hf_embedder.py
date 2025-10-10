"""
HuggingFace Embedder Base
=========================
Shared logic for local SentenceTransformer-based embedders.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

from ..i_embedder import IEmbedder
from ..model.embed_request import EmbedRequest
from ..model.embedding_result import EmbeddingResult

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore


class HFEmbedder(IEmbedder):
    """
    Base class for HuggingFace/SentenceTransformer embedders.
    Single Responsibility: manage model loading and embedding utilities.
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        device: Optional[str] = None,
        normalize: bool = True,
        embedding_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = device or "cpu"
        self.normalize = normalize
        self._model: Optional[SentenceTransformer] = None
        self._dimensions: Optional[int] = None
        self._embedding_fn = embedding_fn

    def _ensure_model(self):
        """Lazily load SentenceTransformer model."""
        if self._embedding_fn:
            return
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is required for HFEmbedder when embedding_fn is not provided."
                )
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimensions = self._model.get_sentence_embedding_dimension()

    def get_dimensions(self) -> int:
        if self._dimensions is not None:
            return self._dimensions
        if self._embedding_fn:
            raise ValueError("Embedding dimensions unknown when using custom embedding_fn without metadata.")
        self._ensure_model()
        assert self._model is not None
        self._dimensions = self._model.get_sentence_embedding_dimension()
        return self._dimensions

    def _run_embedding(self, texts: Sequence[str]) -> List[List[float]]:
        """Execute embedding and return list of vectors."""
        if not texts:
            return []
        if self._embedding_fn:
            vectors = self._embedding_fn(texts)
        else:
            self._ensure_model()
            assert self._model is not None
            vectors = self._model.encode(texts, normalize_embeddings=self.normalize)
        # Convert to list of floats to ensure JSON serializable output
        if np is not None and hasattr(vectors, "tolist"):
            return vectors.tolist()
        return [list(vec) for vec in vectors]

    def embed(self, text: str) -> List[float]:
        return self._run_embedding([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return self._run_embedding(texts)

    def embed_request(self, req: EmbedRequest) -> EmbeddingResult:
        vector = self.embed(req.text)
        return self._build_result(req, vector)

    def embed_batch_req(self, reqs: Iterable[EmbedRequest]) -> List[EmbeddingResult]:
        req_list = list(reqs)
        if not req_list:
            return []
        vectors = self.embed_batch([req.text for req in req_list])
        return [self._build_result(req, vec) for req, vec in zip(req_list, vectors)]

    def _build_result(self, req: EmbedRequest, vector: List[float]) -> EmbeddingResult:
        """Assemble embedding result with metadata."""
        token_count = req.tokens_estimate or len(req.text) // 4
        metadata = req.metadata.copy()
        metadata.update(
            {
                "is_table": req.is_table,
                "title": req.title,
                "section_path": req.section_path,
            }
        )
        return EmbeddingResult(
            chunk_id=req.chunk_id,
            embedding=vector,
            text_embedded=req.text,
            token_count=token_count,
            model_name=self.model_name,
            metadata=metadata,
            doc_id=req.doc_id,
            lang=req.lang,
        )

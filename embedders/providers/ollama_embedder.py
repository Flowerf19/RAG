"""
Ollama Embedder
===============
Embedder implementation that delegates to a local Ollama instance.
"""

from __future__ import annotations

import json
import math
from typing import Iterable, List, Optional, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..i_embedder import IEmbedder
from ..model.embed_request import EmbedRequest
from ..model.embedding_result import EmbeddingResult


class OllamaEmbedder(IEmbedder):
    """
    Embedder backed by Ollama's embeddings API.
    Single Responsibility: marshall EmbedRequest objects to Ollama and
    transform the results into EmbeddingResult instances.
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 8192,
        normalize: bool = True,
        endpoint: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.normalize = normalize
        self.device = "local"
        self._endpoint = (endpoint or "http://127.0.0.1:11434").rstrip("/") + "/api/embeddings"
        self._timeout = timeout
        self._dimensions: Optional[int] = None

    # ------------------------------------------------------------------
    # IEmbedder interface
    # ------------------------------------------------------------------
    def get_dimensions(self) -> int:
        if self._dimensions is None:
            # Trigger a probe embedding to capture dimension lazily.
            probe_vector = self.embed("dimension probe text")
            self._dimensions = len(probe_vector)
        return self._dimensions

    def embed(self, text: str) -> List[float]:
        vector = self._invoke_ollama(text)
        if self.normalize:
            vector = self._l2_normalize(vector)
        if self._dimensions is None:
            self._dimensions = len(vector)
        return vector

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]

    def embed_request(self, req: EmbedRequest) -> EmbeddingResult:
        vector = self.embed(req.text)
        return self._build_result(req, vector)

    def embed_batch_req(self, reqs: Iterable[EmbedRequest]) -> List[EmbeddingResult]:
        results: List[EmbeddingResult] = []
        for req in reqs:
            vector = self.embed(req.text)
            results.append(self._build_result(req, vector))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _invoke_ollama(self, text: str) -> List[float]:
        payload = json.dumps({"model": self.model_name, "prompt": text}).encode("utf-8")
        request = urllib_request.Request(
            self._endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Failed to reach Ollama embeddings endpoint: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON payload from Ollama embeddings endpoint: {raw}") from exc

        if "error" in parsed:
            raise RuntimeError(f"Ollama error: {parsed['error']}")

        embedding = parsed.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Ollama embeddings response missing 'embedding': {parsed}")

        try:
            return [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Ollama returned non-numeric embedding values: {embedding}") from exc

    @staticmethod
    def _l2_normalize(vector: Sequence[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return list(vector)
        return [value / norm for value in vector]

    def _build_result(self, req: EmbedRequest, vector: Sequence[float]) -> EmbeddingResult:
        token_count = req.tokens_estimate or max(1, len(req.text) // 4)
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
            embedding=list(vector),
            text_embedded=req.text,
            token_count=token_count,
            model_name=self.model_name,
            metadata=metadata,
            doc_id=req.doc_id,
            lang=req.lang,
        )

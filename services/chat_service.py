from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from llm.chat_handler import build_messages
from llm.client_factory import LLMClientFactory
from llm.config_loader import ui_default_backend

from services.rag_service import RAGService

logger = logging.getLogger(__name__)


class ChatService:
    """
    High-level chat orchestration helper.

    Uses RAGService to fetch contextual chunks, builds prompts via chat_handler,
    and routes the request to the configured LLM provider.
    """

    def __init__(
        self,
        rag_service: RAGService,
        *,
        default_provider: Optional[str] = None,
    ) -> None:
        self.rag_service = rag_service
        self.default_provider = (default_provider or ui_default_backend()).lower()

    def run_chat(
        self,
        query: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
        provider: Optional[str] = None,
        top_k: int = 3,
        include_sources: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate an answer for the provided query.

        Args:
            query: User question.
            history: Optional chat history.
            provider: Provider override (defaults to config/ui.default_backend).
            top_k: Number of contextual chunks to retrieve.
            include_sources: Whether to include chunk metadata in response.
            temperature: Optional sampling override.
            max_tokens: Optional max tokens override.
        """
        if not query or not query.strip():
            raise ValueError("Query text is required.")

        provider_name = (provider or self.default_provider).lower()
        sources = self._retrieve_sources(query, top_k)
        context_text = self._format_context(sources)
        messages = build_messages(query=query, context=context_text, history=history)

        client = LLMClientFactory.create_from_string(provider_name)
        start = time.perf_counter()
        answer = client.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        response: Dict[str, Any] = {
            "answer": answer,
            "provider": provider_name,
            "latency_ms": latency_ms,
        }

        if include_sources:
            response["sources"] = self._serialize_sources(sources)

        return response

    # ------------------------------------------------------------------#
    # Helpers
    # ------------------------------------------------------------------#
    def _retrieve_sources(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        try:
            return self.rag_service.search_similar(query=query, top_k=top_k)
        except Exception as exc:
            logger.warning("Similarity search failed, continuing without context: %s", exc)
            return []

    @staticmethod
    def _format_context(chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "(Không tìm thấy tài liệu liên quan trong kho RAG.)"

        formatted_segments = []
        for chunk in chunks:
            segment = [
                f"Tài liệu: {chunk.get('source_file') or chunk.get('file_name')}",
                f"Trang: {chunk.get('page_number')}",
                "Nội dung:",
                chunk.get("text", ""),
            ]
            formatted_segments.append("\n".join(filter(None, segment)))
        return "\n\n---\n\n".join(formatted_segments)

    @staticmethod
    def _serialize_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        for chunk in chunks:
            serialized.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "score": chunk.get("similarity_score"),
                    "text": chunk.get("text", ""),
                    "metadata": {
                        "document": chunk.get("source_file") or chunk.get("file_name"),
                        "page_number": chunk.get("page_number"),
                        "token_count": chunk.get("token_count"),
                        "chunk_index": chunk.get("chunk_index"),
                        "block_type": chunk.get("block_type"),
                    },
                }
            )
        return serialized

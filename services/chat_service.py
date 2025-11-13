from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

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
        documents: Optional[List[str]] = None,
        provider: Optional[str] = None,
        top_k: int = 3,
        include_sources: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retrieval_config: Optional[Dict[str, Any]] = None,
        reranker_tokens: Optional[Dict[str, str]] = None,
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
            retrieval_config: Advanced retrieval configuration (embedder, reranker, etc.).
            reranker_tokens: Optional API tokens for rerankers.
        """
        if not query or not query.strip():
            raise ValueError("Query text is required.")

        retrieval_settings = self._prepare_retrieval_config(retrieval_config, top_k)
        provider_name = (provider or self.default_provider).lower()

        retrieval_result: Dict[str, Any] = {
            "mode": "legacy",
            "success": False,
            "context": "",
            "sources": [],
            "queries": [],
            "retrieval_info": {},
        }
        context_text = ""
        chunk_sources: List[Dict[str, Any]] = []
        fallback_used = bool(documents)
        advanced_allowed = not documents and retrieval_settings["top_k"] > 0

        if advanced_allowed:
            retrieval_result = self.rag_service.retrieve_with_features(
                query=query,
                top_k=retrieval_settings["top_k"],
                max_context_chars=retrieval_settings["max_context_chars"],
                embedder_type=retrieval_settings["embedder_type"],
                reranker_type=retrieval_settings["reranker_type"],
                use_query_enhancement=retrieval_settings["use_query_enhancement"],
                api_tokens=reranker_tokens,
            )
            context_text = retrieval_result.get("context") or ""

        if documents or not retrieval_result.get("success"):
            fallback_used = True
            chunk_sources = self._retrieve_sources(
                query=query,
                top_k=retrieval_settings["top_k"],
                documents=documents,
            )
            if not context_text:
                context_text = self._format_context(chunk_sources)

        if not context_text:
            context_text = self._format_context([])

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
            if not fallback_used and retrieval_result.get("success"):
                response["sources"] = retrieval_result.get("sources", [])
            else:
                response["sources"] = self._serialize_sources(chunk_sources)

        retrieval_info = dict(retrieval_result.get("retrieval_info") or {})
        retrieval_info.setdefault("embedder", retrieval_settings["embedder_type"])
        retrieval_info.setdefault("reranker", retrieval_settings["reranker_type"])
        retrieval_info.setdefault("query_enhanced", retrieval_settings["use_query_enhancement"])
        retrieval_info["fallback"] = fallback_used
        if documents:
            retrieval_info["documents"] = documents

        response["retrieval"] = {
            "mode": retrieval_result.get("mode", "legacy") if not fallback_used else "legacy",
            "queries": retrieval_result.get("queries") or [],
            "info": retrieval_info,
        }

        return response

    # ------------------------------------------------------------------#
    # Helpers
    # ------------------------------------------------------------------#
    def _retrieve_sources(
        self,
        query: str,
        top_k: int,
        documents: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        try:
            return self.rag_service.search_similar(
                query=query,
                top_k=top_k,
                documents=documents,
            )
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

    def _prepare_retrieval_config(
        self,
        config: Optional[Dict[str, Any]],
        fallback_top_k: int,
    ) -> Dict[str, Any]:
        base = {
            "top_k": fallback_top_k,
            "max_context_chars": 8000,
            "embedder_type": "huggingface_local",
            "reranker_type": "bge_m3_hf_local",
            "use_query_enhancement": True,
        }
        if config:
            for key, value in config.items():
                if value is not None:
                    base[key] = value

        default_top_k = fallback_top_k if fallback_top_k is not None else 3
        base["top_k"] = self._safe_int(base.get("top_k"), default=default_top_k, min_value=0)
        base["max_context_chars"] = self._safe_int(base.get("max_context_chars"), default=8000, min_value=500)
        base["embedder_type"] = str(base.get("embedder_type") or "huggingface_local").lower()
        base["reranker_type"] = str(base.get("reranker_type") or "none").lower()
        base["use_query_enhancement"] = bool(base.get("use_query_enhancement", True))
        return base

    @staticmethod
    def _safe_int(value: Any, *, default: int, min_value: int) -> int:
        try:
            number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            number = default
        return number if number >= min_value else min_value

# -*- coding: utf-8 -*-
"""
BM25 ingest manager.

Chịu trách nhiệm chuyển đổi các chunk đã xử lý thành tài liệu BM25, trích
keyword bằng spaCy và đồng bộ trạng thái vào Whoosh index. Đồng thời quản
lý cache tránh index trùng lặp chunk.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from .keyword_extractor import KeywordExtractor

logger = logging.getLogger(__name__)


class WhooshIndexerProtocol(Protocol):
    """
    Giao diện tối thiểu mà Whoosh indexer cần hiện thực cho ingest manager.
    """

    def upsert_documents(self, documents: Iterable["BM25Document"]) -> int: ...
    def delete_documents(self, document_ids: Iterable[str]) -> int: ...


@dataclass(frozen=True)
class BM25Document:
    """
    Gói dữ liệu tối thiểu để upsert vào BM25 index.
    """

    document_id: str
    content: str
    keywords: List[str]
    metadata: Dict[str, Any]


class BM25IngestManager:
    """
    Điều phối ingest BM25 và quản lý cache chunk đã xử lý.
    """

    def __init__(
        self,
        indexer: WhooshIndexerProtocol,
        cache_path: Path,
        *,
        keyword_extractor: Optional[KeywordExtractor] = None,
    ) -> None:
        self.indexer = indexer
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.keyword_extractor = keyword_extractor or KeywordExtractor()
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()
        logger.debug("BM25IngestManager initialised. Cache size=%d", len(self._cache))

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            logger.warning("Chunk cache file malformed, resetting: %s", self.cache_path)
            return {}
        except json.JSONDecodeError:
            logger.warning("Chunk cache file corrupted, resetting: %s", self.cache_path)
            return {}

    def _persist_cache(self) -> None:
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def is_chunk_processed(self, chunk_id: str, content_hash: str) -> bool:
        """
        Kiểm tra chunk đã được xử lý và không thay đổi nội dung hay chưa.
        """
        cached = self._cache.get(chunk_id)
        if not cached:
            return False
        return cached.get("content_hash") == content_hash

    def mark_chunk_processed(
        self,
        chunk_id: str,
        content_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Ghi nhận chunk đã được index thành công.
        """
        self._cache[chunk_id] = {
            "content_hash": content_hash,
            "metadata": metadata or {},
        }
        self._persist_cache()

    def remove_chunk(self, chunk_id: str) -> None:
        """
        Xóa chunk khỏi cache và index.
        """
        if chunk_id in self._cache:
            self._cache.pop(chunk_id, None)
            self._persist_cache()
        removed = self.indexer.delete_documents([chunk_id])
        logger.debug("Removed %d document(s) from BM25 index for chunk_id=%s", removed, chunk_id)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def ingest_chunk_set(
        self,
        chunk_set: Any,
        *,
        lang: Optional[str] = None,
        force: bool = False,
    ) -> int:
        """
        Nhận chunk set từ pipeline và upsert vào BM25 index.

        Args:
            chunk_set: Đối tượng chunkers.model.chunk_set.ChunkSet hoặc tương đương.
            lang: Ngôn ngữ ưu tiên cho keyword extractor (None -> auto detect).
            force: Bỏ qua cache và ép index lại toàn bộ chunk.
        """
        chunks = getattr(chunk_set, "chunks", None)
        if not chunks:
            logger.info("Chunk set rỗng, bỏ qua ingest BM25 (doc_id=%s)", getattr(chunk_set, "doc_id", "unknown"))
            return 0

        documents: List[BM25Document] = []
        pending_cache_updates: List[tuple[str, str, Dict[str, Any]]] = []
        skipped = 0

        for chunk in chunks:
            chunk_id = getattr(chunk, "chunk_id", None)
            text = getattr(chunk, "text", "") or ""
            if not chunk_id or not text.strip():
                skipped += 1
                continue

            content_hash = getattr(chunk, "content_hash", None)
            if not content_hash:
                content_hash = self._calculate_hash(text)

            if not force and self.is_chunk_processed(chunk_id, content_hash):
                logger.debug("Skip existing chunk_id=%s (hash unchanged)", chunk_id)
                skipped += 1
                continue

            chunk_lang = lang or getattr(chunk, "metadata", {}).get("language")
            keywords = self.keyword_extractor.extract_keywords(text, chunk_lang)

            metadata = self._build_metadata(chunk_set, chunk, extra={"keywords": keywords})
            documents.append(
                BM25Document(
                    document_id=chunk_id,
                    content=text,
                    keywords=keywords,
                    metadata=metadata,
                )
            )
            pending_cache_updates.append((chunk_id, content_hash, metadata))

        if not documents:
            return 0

        inserted = self.indexer.upsert_documents(documents)
        for chunk_id, content_hash, metadata in pending_cache_updates:
            self.mark_chunk_processed(chunk_id, content_hash, metadata)

        logger.info(
            "BM25 ingest complete for doc_id=%s, indexed=%d, skipped=%d",
            getattr(chunk_set, "doc_id", "unknown"),
            inserted,
            skipped,
        )
        return inserted

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _build_metadata(self, chunk_set: Any, chunk: Any, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "doc_id": getattr(chunk_set, "doc_id", None),
            "chunk_id": getattr(chunk, "chunk_id", None),
            "section_title": getattr(chunk, "section_title", None),
            "source_path": getattr(chunk_set, "file_path", None),
        }

        provenance = getattr(chunk, "provenance", None)
        if provenance:
            metadata["page_numbers"] = sorted(getattr(provenance, "page_numbers", []))
            metadata["source_blocks"] = getattr(provenance, "source_blocks", [])

        metadata.update(getattr(chunk, "metadata", {}) or {})
        if extra:
            metadata.update(extra)
        return metadata

    def _calculate_hash(self, text: str) -> str:
        """
        Hash helper (md5) dùng khi chunk chưa cung cấp sẵn content hash.
        """
        import hashlib

        return hashlib.md5(text.encode("utf-8")).hexdigest()

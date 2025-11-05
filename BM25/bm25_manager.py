"""
BM25 Manager Module
===================
Handles BM25 indexing and search operations for the RAG pipeline.
Single Responsibility: BM25 index management and keyword-based retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from BM25.ingest_manager import BM25IngestManager
    from BM25.whoosh_indexer import WhooshIndexer
    from BM25.search_service import BM25SearchService
    _BM25_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    BM25IngestManager = None  # type: ignore[assignment]
    WhooshIndexer = None  # type: ignore[assignment]
    BM25SearchService = None  # type: ignore[assignment]
    _BM25_IMPORT_ERROR = exc


class BM25Manager:
    """
    Manages BM25 indexing and search for keyword-based retrieval.
    Gracefully handles missing BM25 dependencies.
    """

    def __init__(self, output_dir: Path, cache_dir: Path):
        """
        Initialize BM25 manager.
        
        Args:
            output_dir: Root output directory
            cache_dir: Cache directory for BM25 chunk cache
        """
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        
        # BM25 components (may be None if dependencies unavailable)
        self.ingest_manager: Optional[Any] = None
        self.indexer: Optional[Any] = None
        self.search_service: Optional[Any] = None
        
        self._setup_components()

    def _setup_components(self) -> None:
        """
        Initialize BM25 index infrastructure if dependencies are available.
        """
        if _BM25_IMPORT_ERROR:
            logger.warning("BM25 components unavailable, skipping BM25 ingest: %s", _BM25_IMPORT_ERROR)
            return

        try:
            bm25_index_dir = self.output_dir / "bm25_index"
            bm25_index_dir.mkdir(parents=True, exist_ok=True)
            bm25_cache_file = self.cache_dir / "bm25_chunk_cache.json"
            
            self.indexer = WhooshIndexer(bm25_index_dir)  # type: ignore
            self.ingest_manager = BM25IngestManager(
                indexer=self.indexer,
                cache_path=bm25_cache_file,
            )  # type: ignore
            self.search_service = BM25SearchService(self.indexer)  # type: ignore
            
            logger.info("BM25 manager initialized (index dir=%s)", bm25_index_dir)
        except Exception as exc:
            logger.warning("Failed to initialize BM25 components: %s", exc)
            self.ingest_manager = None
            self.indexer = None
            self.search_service = None

    def ingest_chunk_set(self, chunk_set: Any) -> int:
        """
        Push chunk set into BM25 index; failures should not break the pipeline.
        
        Args:
            chunk_set: ChunkSet object from chunker
            
        Returns:
            Number of chunks indexed (0 if failed or BM25 unavailable)
        """
        if not self.ingest_manager:
            return 0

        try:
            indexed = self.ingest_manager.ingest_chunk_set(chunk_set)
            logger.info("BM25 ingest indexed %d chunk(s)", indexed)
            return indexed
        except Exception as exc:
            logger.warning("BM25 ingest failed, continuing without BM25 index: %s", exc)
            return 0

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        normalize_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute BM25 search and return results in dictionary form.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            normalize_scores: Whether to normalize scores
            
        Returns:
            List of result dictionaries with metadata
        """
        if not self.search_service:
            return []
            
        try:
            results = self.search_service.search(
                query,
                top_k=top_k,
                normalize_scores=normalize_scores,
            )
        except Exception as exc:
            logger.warning("BM25 search failed: %s", exc)
            return []

        # Format results for consistency with FAISS retrieval
        formatted_results: List[Dict[str, Any]] = []
        for result in results:
            metadata = result.metadata or {}
            source_path = metadata.get("source_path")
            file_name = metadata.get("file_name")
            
            if not file_name and source_path:
                file_name = Path(source_path).name
                
            page_numbers = metadata.get("page_numbers") or []
            primary_page = None
            if page_numbers:
                primary_page = page_numbers[0]
            else:
                primary_page = metadata.get("page_number")
                
            formatted_results.append({
                "chunk_id": result.document_id,
                "bm25_raw_score": result.raw_score,
                "bm25_normalized_score": result.normalized_score,
                "keywords": metadata.get("keywords", []),
                "text": metadata.get("text", ""),
                "file_name": file_name,
                "source_path": source_path,
                "page_number": primary_page,
                "page_numbers": page_numbers,
                "metadata": metadata,
            })
            
        return formatted_results

    def is_available(self) -> bool:
        """Check if BM25 components are available and initialized."""
        return self.search_service is not None

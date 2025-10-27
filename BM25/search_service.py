# -*- coding: utf-8 -*-
"""
High-level BM25 search service coordinating keyword extraction and Whoosh queries.

The service is responsible for:
    * chuẩn hóa truy vấn bằng spaCy keyword extractor
    * chuyển sang Whoosh indexer để lấy kết quả BM25
    * chuẩn hóa điểm số (z-score) khi cần, kèm safeguard cho tập kết quả nhỏ
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence
from .keyword_extractor import KeywordExtractor

logger = logging.getLogger(__name__)


class IndexBackend(Protocol):
    """
    Minimal giao diện mà Whoosh indexer (hay bất kỳ backend nào) cần hiện thực.
    """

    def search(
        self, terms: Sequence[str], *, limit: int = 10
    ) -> List["IndexerHit"]: ...


@dataclass(frozen=True)
class IndexerHit:
    """
    Định dạng kết quả trả về từ lớp indexer mức thấp.
    """

    document_id: str
    score: float
    metadata: Optional[dict] = None


@dataclass(frozen=True)
class SearchResult:
    """
    Định dạng kết quả cuối cùng sau khi dịch vụ xử lý.
    """

    document_id: str
    raw_score: float
    normalized_score: float
    keywords: List[str]
    metadata: Optional[dict] = None


class BM25SearchService:
    """
    Bọc logic tìm kiếm BM25 với chuẩn hóa điểm số và extract keyword.
    """

    def __init__(
        self,
        index_backend: IndexBackend,
        keyword_extractor: Optional[KeywordExtractor] = None,
        *,
        default_top_k: int = 10,
        min_terms: int = 1,
        zscore_min_count: int = 5,
    ) -> None:
        self.index_backend = index_backend
        self.keyword_extractor = keyword_extractor or KeywordExtractor()
        self.default_top_k = default_top_k
        self.min_terms = min_terms
        self.zscore_min_count = zscore_min_count

    def search(
        self,
        query: str,
        *,
        lang: Optional[str] = None,
        top_k: Optional[int] = None,
        normalize_scores: bool = True,
    ) -> List[SearchResult]:
        """
        Thực hiện truy vấn BM25 với keyword extractor spaCy.
        """
        if not query:
            logger.warning("BM25 search called with empty query")
            return []

        terms = self.keyword_extractor.extract_keywords(query, lang, max_terms=50)
        if len(terms) < self.min_terms:
            logger.info(
                "Không đủ từ khóa (terms=%d). Fallback sang truy vấn thô.", len(terms)
            )
            terms = [query]

        limit = top_k or self.default_top_k
        logger.debug("BM25 search terms=%s limit=%s", terms, limit)
        hits = self.index_backend.search(terms, limit=limit)

        if not hits:
            return []

        raw_results = [
            SearchResult(
                document_id=hit.document_id,
                raw_score=hit.score,
                normalized_score=hit.score,  # tạm gán, có thể chỉnh dưới
                keywords=terms,
                metadata=hit.metadata,
            )
            for hit in hits
        ]

        if normalize_scores:
            return self._apply_normalization(raw_results)

        return raw_results

    def _apply_normalization(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Chuẩn hóa điểm BM25 bằng z-score nếu đủ mẫu, ngược lại giữ nguyên.
        """
        if len(results) < self.zscore_min_count:
            logger.debug(
                "Skip z-score normalization (result count %d < %d)",
                len(results),
                self.zscore_min_count,
            )
            return results

        scores = [result.raw_score for result in results]
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            logger.debug("Skip z-score normalization due to zero std deviation")
            return results

        normalized: List[SearchResult] = []
        for result in results:
            z_score = (result.raw_score - mean) / std_dev
            normalized.append(
                SearchResult(
                    document_id=result.document_id,
                    raw_score=result.raw_score,
                    normalized_score=z_score,
                    keywords=result.keywords,
                    metadata=result.metadata,
                )
            )
        return normalized

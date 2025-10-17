# -*- coding: utf-8 -*-
"""
Whoosh index backend dùng cho pipeline BM25.

Hiện thực giao diện `IndexBackend` mà `BM25SearchService` mong đợi, đồng thời
phù hợp với `BM25IngestManager` cho các thao tác upsert/delete.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:
    from whoosh import index
    from whoosh.analysis import RegexTokenizer, LowercaseFilter, StopFilter
    from whoosh.fields import ID, KEYWORD, Schema, STORED, TEXT
    from whoosh.qparser import MultifieldParser, OrGroup
    from whoosh.scoring import BM25F
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "Whoosh chưa được cài đặt. Vui lòng thêm whoosh vào requirements và pip install."
    ) from exc

from .ingest_manager import BM25Document
from .search_service import IndexerHit

logger = logging.getLogger(__name__)


def default_analyzer():
    """
    Tạo analyzer mặc định (tokenize + lowercase + stopword filter nhẹ).
    """
    return RegexTokenizer() | LowercaseFilter() | StopFilter()


class WhooshIndexer:
    """
    Bao bọc Whoosh index với các thao tác insert/search phù hợp cho BM25.
    """

    def __init__(
        self,
        index_dir: Path | str,
        *,
        recreate: bool = False,
    ) -> None:
        self.index_path = Path(index_dir)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.schema = Schema(
            document_id=ID(stored=True, unique=True),
            content=TEXT(analyzer=default_analyzer(), stored=False),
            keywords=KEYWORD(lowercase=True, commas=True, scorable=True, stored=True),
            metadata=STORED,
        )

        if recreate and self.index_path.exists():
            for child in self.index_path.glob("*"):
                if child.is_file():
                    child.unlink()

        if index.exists_in(self.index_path):
            self._index = index.open_dir(self.index_path)
        else:
            self._index = index.create_in(self.index_path, self.schema)

        self.parser = MultifieldParser(
            ["keywords", "content"],
            schema=self.schema,
            group=OrGroup.factory(0.5),
        )
        self.scoring = BM25F()

    # ------------------------------------------------------------------
    # Upsert/Delete
    # ------------------------------------------------------------------
    def upsert_documents(self, documents: Iterable[BM25Document]) -> int:
        """
        Thêm/cập nhật tài liệu vào index.
        """
        docs = list(documents)
        if not docs:
            return 0

        writer = self._index.writer()
        for doc in docs:
            keyword_field = ",".join(doc.keywords)
            writer.update_document(
                document_id=doc.document_id,
                content=doc.content,
                keywords=keyword_field,
                metadata=json.dumps(doc.metadata, ensure_ascii=False),
            )
        writer.commit()
        logger.debug("Whoosh upsert %d document(s)", len(docs))
        return len(docs)

    def delete_documents(self, document_ids: Iterable[str]) -> int:
        ids = [doc_id for doc_id in document_ids if doc_id]
        if not ids:
            return 0

        writer = self._index.writer()
        for doc_id in ids:
            writer.delete_by_term("document_id", doc_id)
        writer.commit()
        logger.debug("Whoosh deleted %d document(s)", len(ids))
        return len(ids)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, terms: Sequence[str], *, limit: int = 10) -> List[IndexerHit]:
        """
        Thực thi truy vấn BM25 trên Whoosh và trả về danh sách IndexerHit.
        """
        if not terms:
            return []

        query_text = " ".join(terms)
        query = self.parser.parse(query_text)

        with self._index.searcher(weighting=self.scoring) as searcher:
            results = searcher.search(query, limit=limit)
            hits: List[IndexerHit] = []
            for hit in results:
                metadata_serialized: Optional[str] = hit.get("metadata")
                metadata = None
                if metadata_serialized:
                    try:
                        metadata = json.loads(metadata_serialized)
                    except json.JSONDecodeError:
                        metadata = {"raw_metadata": metadata_serialized}

                hits.append(
                    IndexerHit(
                        document_id=hit["document_id"],
                        score=float(hit.score),
                        metadata=metadata,
                    )
                )
        logger.debug("Whoosh search terms=%s -> %d hits", terms, len(hits))
        return hits


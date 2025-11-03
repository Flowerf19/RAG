"""
Block level models representing the atomic units coming out of the loader.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .base import LoaderBaseModel
from .table import TableSchema


BBox = Tuple[float, float, float, float]


def _compute_hash(text: str, block_type: str) -> str:
    payload = f"{block_type}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class Block(LoaderBaseModel):
    block_id: str
    page_number: int
    text: str
    bbox: BBox
    block_type: str = "text"
    category: Optional[str] = None
    score: Optional[float] = None
    stable_id: Optional[str] = None
    text_source: str = "pdf_extract_kit"
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if self.stable_id is None:
            self.stable_id = self.block_id

        if self.content_sha256 is None:
            normalized = (self.text or "").strip()
            self.content_sha256 = _compute_hash(normalized, self.block_type)

    def normalized(self) -> Optional["Block"]:
        normalized_text = (self.text or "").strip()
        if not normalized_text and self.block_type not in {"table", "figure", "image", "formula"}:
            return None

        return Block(
            block_id=self.block_id,
            page_number=self.page_number,
            text=normalized_text,
            bbox=self.bbox,
            block_type=self.block_type,
            category=self.category,
            score=self.score,
            stable_id=self.stable_id,
            text_source=self.text_source,
            metadata=dict(self.metadata or {}),
            content_sha256=self.content_sha256,
        )


@dataclass
class TableBlock(Block):
    table: Optional[TableSchema] = None

    def normalized(self) -> Optional["TableBlock"]:
        normalized_text = (self.text or "").strip()
        if not normalized_text and (self.table is None or self.table.is_empty()):
            return None

        metadata = dict(self.metadata or {})
        if self.table and "table_schema" not in metadata:
            metadata["table_schema"] = self.table

        return TableBlock(
            block_id=self.block_id,
            page_number=self.page_number,
            text=normalized_text,
            bbox=self.bbox,
            block_type=self.block_type,
            category=self.category,
            score=self.score,
            stable_id=self.stable_id,
            text_source=self.text_source,
            metadata=metadata,
            content_sha256=self.content_sha256 or _compute_hash(normalized_text, "table"),
            table=self.table,
        )

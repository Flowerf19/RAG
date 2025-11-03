"""
Document level model returned by the loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from .base import LoaderBaseModel
from .page import PDFPage


@dataclass
class PDFDocument(LoaderBaseModel):
    file_path: str
    pages: List[PDFPage]
    meta: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def normalize(self) -> "PDFDocument":
        normalized_pages: List[PDFPage] = []

        for page in self.pages:
            normalized_blocks = []
            seen_hashes = set()
            for block in page.blocks:
                normalized_block = block.normalized()
                if not normalized_block:
                    continue

                if normalized_block.content_sha256:
                    if normalized_block.content_sha256 in seen_hashes:
                        continue
                    seen_hashes.add(normalized_block.content_sha256)
                normalized_blocks.append(normalized_block)

            normalized_pages.append(
                page.with_blocks(
                    normalized_blocks,
                    tables=list(page.tables),
                    figures=list(page.figures),
                )
            )

        meta = dict(self.meta)
        meta["normalized_at"] = datetime.now(timezone.utc).isoformat()
        meta["normalized"] = True

        return PDFDocument(
            file_path=self.file_path,
            pages=normalized_pages,
            meta=meta,
            warnings=list(self.warnings),
        )

    @property
    def page_count(self) -> int:
        return len(self.pages)

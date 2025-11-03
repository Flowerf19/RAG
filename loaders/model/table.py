"""
Table model helpers used by the loader pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .base import LoaderBaseModel


BBox = Tuple[float, float, float, float]


@dataclass
class TableCell(LoaderBaseModel):
    row_index: int
    col_index: int
    value: str
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None


@dataclass
class TableRow(LoaderBaseModel):
    index: int
    cells: List[TableCell] = field(default_factory=list)

    def values(self) -> List[str]:
        return [cell.value for cell in self.cells]


@dataclass
class TableSchema(LoaderBaseModel):
    id: str
    page_number: int
    header: List[str] = field(default_factory=list)
    rows: List[TableRow] = field(default_factory=list)
    bbox: Optional[BBox] = None
    source: str = "pdf_extract_kit"
    markdown: Optional[str] = None

    def build_markdown(self) -> str:
        lines: List[str] = []
        if self.header:
            lines.append(" | ".join(self.header))
            lines.append(" | ".join("---" for _ in self.header))
        for row in self.rows:
            values = [cell.value for cell in row.cells]
            if values:
                lines.append(" | ".join(values))
        markdown = "\n".join(lines)
        self.markdown = markdown
        return markdown

    def embedding_text(self) -> str:
        return self.markdown or self.build_markdown()

    def cell_count(self) -> int:
        return sum(len(row.cells) for row in self.rows)

    def is_empty(self) -> bool:
        if self.header:
            return False
        return self.cell_count() == 0

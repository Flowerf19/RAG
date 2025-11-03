"""
Page level representation built on top of loader blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import LoaderBaseModel
from .block import Block, TableBlock
from .table import TableSchema


@dataclass
class PDFPage(LoaderBaseModel):
    page_number: int
    width: float
    height: float
    blocks: List[Block] = field(default_factory=list)
    tables: List[TableSchema] = field(default_factory=list)
    figures: List[Block] = field(default_factory=list)
    text: str = ""
    source: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def with_blocks(
        self,
        blocks: List[Block],
        *,
        tables: Optional[List[TableSchema]] = None,
        figures: Optional[List[Block]] = None,
    ) -> "PDFPage":
        return PDFPage(
            page_number=self.page_number,
            width=self.width,
            height=self.height,
            blocks=blocks,
            tables=tables if tables is not None else list(self.tables),
            figures=figures if figures is not None else list(self.figures),
            text="\n\n".join(block.text for block in blocks if block.text),
            source=dict(self.source),
            warnings=list(self.warnings),
        )

    def add_block(self, block: Block) -> None:
        self.blocks.append(block)
        if isinstance(block, TableBlock) and block.table:
            self.tables.append(block.table)
        if block.block_type == "figure":
            self.figures.append(block)

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)

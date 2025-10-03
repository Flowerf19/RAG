from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from .base import ChunkerBaseModel


@dataclass
class ChunkMetadata(ChunkerBaseModel):
    """
    Metadata cho một chunk.
    
    Attributes:
        source: Source document path/name
        document_id: ID của document gốc
        page: Page number (nếu có)
        page_range: Range of pages nếu chunk span multiple pages
        section: Section/heading name (nếu có)
        chunk_index: Index của chunk trong document
        total_chunks: Tổng số chunks trong document
        citation: Human-readable citation
        parent_chunk_id: ID của parent chunk (cho hierarchical chunking)
        child_chunk_ids: IDs của child chunks (cho hierarchical chunking)
        bbox: Bounding box (nếu có position info)
        extra: Additional metadata fields
    """
    source: str
    document_id: str
    page: Optional[int] = None
    page_range: Optional[tuple[int, int]] = None
    section: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    citation: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    bbox: Optional[Dict[str, float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate metadata."""
        if not self.source or not self.document_id:
            return False
        if self.page is not None and self.page < 0:
            return False
        if self.chunk_index is not None and self.chunk_index < 0:
            return False
        return True
    
    def add_extra(self, key: str, value: Any):
        """Add extra metadata field."""
        self.extra[key] = value
    
    def get_extra(self, key: str, default: Any = None) -> Any:
        """Get extra metadata field."""
        return self.extra.get(key, default)

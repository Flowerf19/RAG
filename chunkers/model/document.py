from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .base import ChunkerBaseModel
from .chunk import Chunk


@dataclass
class ChunkDocument(ChunkerBaseModel):
    """
    Container cho tất cả chunks của một document.
    
    Attributes:
        document_id: ID của document gốc
        source: Source document path
        chunks: List of chunks
        metadata: Document-level metadata
        total_chunks: Total number of chunks
        chunking_strategy: Strategy used for chunking
        config: Chunking configuration used
    """
    document_id: str
    source: str
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_chunks: int = 0
    chunking_strategy: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-init processing."""
        self.total_chunks = len(self.chunks)
        # Update metadata in chunks
        for i, chunk in enumerate(self.chunks):
            if chunk.metadata.total_chunks is None:
                chunk.metadata.total_chunks = self.total_chunks
            if chunk.metadata.chunk_index is None:
                chunk.metadata.chunk_index = i
    
    def add_chunk(self, chunk: Chunk):
        """Add a chunk to document."""
        self.chunks.append(chunk)
        self.total_chunks = len(self.chunks)
        # Update chunk metadata
        chunk.metadata.total_chunks = self.total_chunks
        if chunk.metadata.chunk_index is None:
            chunk.metadata.chunk_index = self.total_chunks - 1
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by stable_id."""
        for chunk in self.chunks:
            if chunk.stable_id == chunk_id:
                return chunk
        return None
    
    def get_chunk_by_index(self, index: int) -> Optional[Chunk]:
        """Get chunk by index."""
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None
    
    def get_chunks_by_page(self, page: int) -> List[Chunk]:
        """Get all chunks from a specific page."""
        return [
            chunk for chunk in self.chunks
            if chunk.metadata.page == page
        ]
    
    def get_chunks_by_section(self, section: str) -> List[Chunk]:
        """Get all chunks from a specific section."""
        return [
            chunk for chunk in self.chunks
            if chunk.metadata.section == section
        ]
    
    def validate(self) -> bool:
        """Validate all chunks."""
        if not self.document_id or not self.source:
            return False
        if self.total_chunks != len(self.chunks):
            return False
        for chunk in self.chunks:
            if not chunk.validate():
                return False
        return True
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute statistics về chunks."""
        if not self.chunks:
            return {
                "total_chunks": 0,
                "total_chars": 0,
                "total_tokens": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }
        
        char_counts = [chunk.char_count or 0 for chunk in self.chunks]
        token_counts = [chunk.token_count or 0 for chunk in self.chunks]
        
        return {
            "total_chunks": self.total_chunks,
            "total_chars": sum(char_counts),
            "total_tokens": sum(token_counts) if any(token_counts) else None,
            "avg_chunk_size": sum(char_counts) // len(char_counts) if char_counts else 0,
            "min_chunk_size": min(char_counts) if char_counts else 0,
            "max_chunk_size": max(char_counts) if char_counts else 0,
            "content_types": self._count_content_types(),
        }
    
    def _count_content_types(self) -> Dict[str, int]:
        """Count chunks by content type."""
        counts = {"text": 0, "table": 0, "hybrid": 0}
        for chunk in self.chunks:
            counts[chunk.content_type] = counts.get(chunk.content_type, 0) + 1
        return counts
    
    def to_dict(self) -> dict:
        """Convert to dict."""
        data = super().to_dict()
        # Convert chunks to dicts
        data['chunks'] = [chunk.to_dict() for chunk in self.chunks]
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChunkDocument':
        """Create from dict."""
        chunks_data = data.pop('chunks', [])
        chunks = [Chunk.from_dict(c) for c in chunks_data]
        doc = cls(**data)
        doc.chunks = chunks
        return doc

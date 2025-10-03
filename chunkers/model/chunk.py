from dataclasses import dataclass, field
from typing import Optional, List, Literal
from .base import ChunkerBaseModel
from .metadata import ChunkMetadata


@dataclass
class Chunk(ChunkerBaseModel):
    """
    Một chunk của document.
    
    Attributes:
        stable_id: Deterministic ID (hash-based)
        content: Text content của chunk
        content_type: Loại content (text, table, hybrid)
        metadata: Chunk metadata
        token_count: Số tokens (nếu đã tính)
        char_count: Số characters
        embedding: Vector embedding (nếu đã generate)
        previous_chunk_id: ID của chunk trước đó
        next_chunk_id: ID của chunk tiếp theo
    """
    stable_id: str
    content: str
    content_type: Literal["text", "table", "hybrid"] = "text"
    metadata: ChunkMetadata = field(default_factory=lambda: ChunkMetadata(source="", document_id=""))
    token_count: Optional[int] = None
    char_count: Optional[int] = None
    embedding: Optional[List[float]] = None
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    
    def __post_init__(self):
        """Post-init processing."""
        if self.char_count is None:
            self.char_count = len(self.content)
    
    def validate(self) -> bool:
        """Validate chunk."""
        if not self.stable_id or not self.content:
            return False
        if not self.metadata.validate():
            return False
        if self.char_count is not None and self.char_count != len(self.content):
            return False
        return True
    
    def get_content_preview(self, max_length: int = 100) -> str:
        """Get content preview (truncated)."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    def compute_token_count(self, tokenizer=None) -> int:
        """
        Compute token count.
        
        Args:
            tokenizer: Tokenizer to use (tiktoken, transformers, etc.)
        
        Returns:
            Token count
        """
        if tokenizer is None:
            # Simple approximation: 1 token ~ 4 chars
            self.token_count = len(self.content) // 4
        else:
            # Use provided tokenizer
            if hasattr(tokenizer, 'encode'):
                self.token_count = len(tokenizer.encode(self.content))
            else:
                self.token_count = len(self.content) // 4
        
        return self.token_count
    
    def to_dict(self) -> dict:
        """Convert to dict, exclude embedding if too large."""
        data = super().to_dict()
        # Optionally exclude embedding from serialization
        if 'embedding' in data and data['embedding'] is not None:
            # Keep embedding info but truncate for display
            data['embedding_dim'] = len(data['embedding'])
            data['embedding'] = None  # or truncate
        return data

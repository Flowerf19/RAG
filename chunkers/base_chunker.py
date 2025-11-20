"""
Base Chunker Interface
======================
Định nghĩa abstract base class cho tất cả các chunker strategies.
Tuân thủ Single Responsibility Principle và Open/Closed Principle.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .model import ChunkSet, Chunk
from .config_loader import get_chunker_config

# Import from new PDFLoaders package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "PDFLoaders"))
try:
    from PDFLoaders.provider import PDFDocument, PageContent
except ImportError:
    from PDFLoaders.provider.models import PDFDocument, PageContent


class BaseChunker(ABC):
    """
    Abstract base class cho tất cả chunking strategies.
    Single Responsibility: Định nghĩa interface chung cho chunkers.
    """
    
    def __init__(self, max_tokens: Optional[int] = None, overlap_tokens: Optional[int] = None, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            max_tokens: Số tokens tối đa cho mỗi chunk (None = load from config)
            overlap_tokens: Số tokens overlap giữa các chunks (None = load from config)
            config: Custom config dict (None = load from YAML)
        """
        # Load config if not provided
        if config is None:
            config = get_chunker_config("default")
        
        # Use provided values or fallback to config
        self.max_tokens = max_tokens if max_tokens is not None else config.get("max_tokens", 500)
        self.overlap_tokens = overlap_tokens if overlap_tokens is not None else config.get("overlap_tokens", 50)
    
    @abstractmethod
    def chunk(self, document: PDFDocument) -> ChunkSet:
        """
        Chunk toàn bộ document thành ChunkSet.
        
        Args:
            document: PDFDocument từ PDFLoaders
            
        Returns:
            ChunkSet chứa tất cả chunks
        """
        pass
    
    def chunk_pages(self, pages: List[PageContent], doc_id: str) -> List[Chunk]:
        """
        Chunk một list pages thành list chunks.
        Default implementation: combine text from all pages.
        
        Args:
            pages: Danh sách PageContent cần chunk
            doc_id: Document ID
            
        Returns:
            List các Chunk objects
        """
        # Default: combine all text from pages
        combined_text = "\n\n".join(page.text for page in pages if page.text.strip())
        
        if not combined_text.strip():
            return []
        
        # Simple split by max_tokens
        chunks = []
        words = combined_text.split()
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = len(word) // 4 + 1  # Rough estimation
            if current_tokens + word_tokens > self.max_tokens and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                    text=chunk_text,
                    doc_id=doc_id,
                    token_count=current_tokens
                ))
                # Overlap
                overlap_words = int(len(current_chunk) * (self.overlap_tokens / self.max_tokens))
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                current_tokens = sum(len(w) // 4 + 1 for w in current_chunk)
            
            current_chunk.append(word)
            current_tokens += word_tokens
        
        # Last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{len(chunks)}",
                text=chunk_text,
                doc_id=doc_id,
                token_count=current_tokens
            ))
        
        return chunks
    
    def validate_config(self) -> bool:
        """Validate chunker configuration"""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be less than max_tokens")
        return True
    
    def estimate_tokens(self, text: str) -> int:
        """
        Ước lượng số tokens trong text.
        Mặc định: ~4 chars = 1 token (có thể override cho chính xác hơn)
        """
        return len(text) // 4

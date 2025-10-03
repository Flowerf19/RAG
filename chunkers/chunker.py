from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import sys
import os

# Add parent directory to path để import loaders
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loaders.model import PDFDocument
from .model import ChunkDocument, Chunk, ChunkMetadata
from .config import ConfigManager, get_chunking_config
from .ids import generate_chunk_id, generate_citation


class BaseChunker(ABC):
    """
    Base class cho tất cả chunking strategies.
    
    Subclasses phải implement method chunk().
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize chunker.
        
        Args:
            config: Chunking configuration (nếu None, load từ YAML)
        """
        if config is None:
            self.config = get_chunking_config(self.strategy_name())
        else:
            self.config = config
        
        self._validate_config()
    
    @classmethod
    @abstractmethod
    def strategy_name(cls) -> str:
        """Return strategy name (e.g., 'fixed_size', 'semantic')."""
        raise NotImplementedError
    
    @abstractmethod
    def chunk(self, pdf_document: PDFDocument) -> ChunkDocument:
        """
        Chunk PDF document thành chunks.
        
        Args:
            pdf_document: PDFDocument from loaders
        
        Returns:
            ChunkDocument containing all chunks
        """
        raise NotImplementedError
    
    def _validate_config(self):
        """Validate configuration. Override trong subclass nếu cần."""
        pass
    
    def _create_chunk_metadata(
        self,
        pdf_document: PDFDocument,
        page: Optional[int] = None,
        chunk_index: Optional[int] = None,
        section: Optional[str] = None,
        **extra
    ) -> ChunkMetadata:
        """
        Create chunk metadata from PDF document.
        
        Args:
            pdf_document: Source PDF document
            page: Page number
            chunk_index: Chunk index
            section: Section name
            **extra: Extra metadata fields
        
        Returns:
            ChunkMetadata
        """
        # Generate citation
        citation = generate_citation(
            document_title=getattr(pdf_document, "title", getattr(pdf_document, "file_path", "")),
            page=page,
            section=section,
            chunk_index=chunk_index
        )
        
        # Unpack extra dict into extra field (avoid nesting dict)
        file_path = str(getattr(pdf_document, 'file_path', '') or '')
        return ChunkMetadata(
            source=file_path,
            document_id=file_path,
            page=page,
            section=section,
            chunk_index=chunk_index,
            citation=citation,
            extra={**extra}
        )
    
    def _create_chunk(
        self,
        content: str,
        pdf_document: PDFDocument,
        content_type: str = "text",
        **metadata_kwargs
    ) -> Chunk:
        """
        Create a chunk.
        
        Args:
            content: Chunk content
            pdf_document: Source PDF document
            content_type: Content type (text, table, hybrid)
            **metadata_kwargs: Metadata fields
        
        Returns:
            Chunk
        """
        metadata = self._create_chunk_metadata(pdf_document, **metadata_kwargs)
        
        # Generate stable ID
        file_path = str(getattr(pdf_document, 'file_path', '') or '')
        stable_id = generate_chunk_id(
            content=content,
            source=file_path,
            page=metadata.page,
            chunk_index=metadata.chunk_index
        )
        # Ensure content_type is Literal
        ct = content_type if content_type in ("text", "table", "hybrid") else "text"
        return Chunk(
            stable_id=stable_id,
            content=content,
            content_type=ct,
            metadata=metadata
        )
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        return self.config.get(key, default)


class FixedSizeChunker(BaseChunker):
    """Fixed size chunking strategy."""
    
    @classmethod
    def strategy_name(cls) -> str:
        return "fixed_size"
    
    def chunk(self, pdf_document: PDFDocument) -> ChunkDocument:
        """Chunk document into fixed-size chunks."""
        chunk_size = self.get_config("chunk_size", 512)
        overlap = self.get_config("overlap", 50)
        unit = self.get_config("unit", "chars")
        
        # TODO: Implement fixed size chunking logic
        # For now, placeholder
        chunks = []
        
        file_path = str(getattr(pdf_document, 'file_path', '') or '')
        chunk_doc = ChunkDocument(
            document_id=file_path,
            source=file_path,
            chunks=chunks,
            chunking_strategy=self.strategy_name(),
            config=self.config
        )
        return chunk_doc


class SemanticChunker(BaseChunker):
    """Semantic chunking strategy."""
    
    @classmethod
    def strategy_name(cls) -> str:
        return "semantic"
    
    def chunk(self, pdf_document: PDFDocument) -> ChunkDocument:
        """Chunk document based on semantics."""
        max_chunk_size = self.get_config("max_chunk_size", 800)
        min_chunk_size = self.get_config("min_chunk_size", 100)
        split_on = self.get_config("split_on", "sentence")
        
        # Sử dụng blocks từ PDFDocument (đã được loader trích xuất)
        chunks = []
        chunk_index = 0
        
        for page_idx, page in enumerate(pdf_document.pages):
            # Trích xuất text từ blocks
            page_text_parts = []
            for block in page.blocks:
                # Blocks từ PyMuPDF là tuple: (x0, y0, x1, y1, text, block_no, block_type)
                if isinstance(block, (tuple, list)) and len(block) >= 5:
                    block_text = block[4]  # Text ở index 4
                    if isinstance(block_text, str) and block_text.strip():
                        page_text_parts.append(block_text.strip())
                # Hoặc là Block object
                elif hasattr(block, 'text') and block.text:
                    page_text_parts.append(block.text.strip())
            
            page_text = "\n".join(page_text_parts)
            
            # Nếu page có text đủ dài, tạo chunk
            if len(page_text.strip()) >= min_chunk_size:
                chunk = self._create_chunk(
                    content=page_text,
                    pdf_document=pdf_document,
                    content_type="text",
                    page=page.page_number,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
        
        file_path = str(getattr(pdf_document, 'file_path', '') or '')
        chunk_doc = ChunkDocument(
            document_id=file_path,
            source=file_path,
            chunks=chunks,
            chunking_strategy=self.strategy_name(),
            config=self.config
        )
        return chunk_doc


class SlidingWindowChunker(BaseChunker):
    """Sliding window chunking strategy."""
    
    @classmethod
    def strategy_name(cls) -> str:
        return "sliding_window"
    
    def chunk(self, pdf_document: PDFDocument) -> ChunkDocument:
        """Chunk document with sliding window."""
        window_size = self.get_config("window_size", 512)
        stride = self.get_config("stride", 256)
        unit = self.get_config("unit", "chars")
        
        # TODO: Implement sliding window chunking logic
        chunks = []
        
        file_path = str(getattr(pdf_document, 'file_path', '') or '')
        chunk_doc = ChunkDocument(
            document_id=file_path,
            source=file_path,
            chunks=chunks,
            chunking_strategy=self.strategy_name(),
            config=self.config
        )
        return chunk_doc

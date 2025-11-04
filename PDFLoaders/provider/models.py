"""
Data models for PDF Provider
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .text_block import TextBlock


@dataclass
class PageContent:
    """
    Content từ 1 page.
    
    Enhanced to support layout-aware processing with TextBlock structure.
    Backward compatible: blocks=None falls back to flat text processing.
    """
    page_number: int
    text: str
    tables: List[List[List[str]]]  # List of tables, mỗi table là 2D array
    figures: List[Dict[str, Any]]  # List of figures with bbox and metadata
    extraction_method: str  # "text", "ocr", "hybrid"
    char_count: int
    language: str = "en"  # Auto-detected language: "en", "zh", etc.
    
    # Enhanced: Layout detection support
    blocks: Optional[List[Any]] = None  # List[TextBlock] if available, else None
    
    @property
    def has_layout_structure(self) -> bool:
        """Check if this page has detected layout structure"""
        return self.blocks is not None and len(self.blocks) > 0
    
    def get_headings(self) -> List[Any]:
        """Extract all heading blocks"""
        if not self.has_layout_structure:
            return []
        assert self.blocks is not None
        return [b for b in self.blocks if b.is_heading()]
    
    def get_content_blocks(self) -> List[Any]:
        """Extract all content (text/paragraph) blocks"""
        if not self.has_layout_structure:
            return []
        assert self.blocks is not None
        return [b for b in self.blocks if b.is_content()]
    
    
@dataclass
class PDFDocument:
    """Document object từ PDF"""
    file_path: str
    total_pages: int
    pages: List[PageContent]
    metadata: Dict[str, Any]

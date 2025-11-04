"""
Simple Text Provider - Basic text extraction without OCR
"""

import fitz
from pathlib import Path
import logging

from .models import PDFDocument, PageContent

logger = logging.getLogger(__name__)


class SimpleTextProvider:
    """
    Simple provider - chỉ extract text, không OCR, không tables
    Dùng cho trường hợp đơn giản hoặc fallback
    """
    
    def __init__(self):
        logger.info("SimpleTextProvider initialized (text-only mode)")
    
    def load(self, pdf_path: str | Path) -> PDFDocument:
        """
        Load PDF với text extraction only
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PDFDocument with text-only content
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "page_count": len(doc),
        }
        
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            page_content = PageContent(
                page_number=page_num + 1,
                text=text,
                tables=[],
                figures=[],  # No figure extraction in simple mode
                extraction_method="text",
                char_count=len(text.strip())
            )
            pages.append(page_content)
        
        doc.close()
        
        return PDFDocument(
            file_path=str(pdf_path),
            total_pages=len(pages),
            pages=pages,
            metadata=metadata
        )

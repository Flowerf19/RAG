"""
PDF Provider using PyMuPDF4LLM for markdown extraction
"""

import fitz
import pymupdf4llm
from pathlib import Path
import logging

from .models import PDFDocument, PageContent
from .extractors import OCRExtractor, FigureExtractor
from .extractors.pymupdf4llm_extractor import PyMuPDF4LLMExtractor

logger = logging.getLogger(__name__)


class PyMuPDF4LLMProvider:
    """
    PDF Provider using PyMuPDF4LLM for markdown extraction
    
    Advantages over pdfplumber:
    - ✅ Preserves captions with tables/figures
    - ✅ Better structure preservation (headings, lists, etc.)
    - ✅ Markdown format ideal for LLMs
    - ✅ No content duplication issues
    - ✅ Built-in table formatting
    
    Logic:
    - Text-based PDF → PyMuPDF4LLM markdown extraction
    - Image-based PDF → PaddleOCR fallback
    - Mixed PDF → Hybrid per page
    - Figures → Group images + OCR text extraction
    """
    
    def __init__(self, 
                 use_ocr: str = "auto",
                 ocr_lang: str = "multilingual",
                 min_text_threshold: int = 50):
        """
        Args:
            use_ocr: "auto", "always", "never"
            ocr_lang: Language for OCR ("multilingual", "en", "ch", "vi")
            min_text_threshold: Minimum chars to consider as text-based page
        """
        self.use_ocr = use_ocr
        self.min_text_threshold = min_text_threshold
        
        # Initialize extractors
        self.ocr_extractor = OCRExtractor(lang=ocr_lang) if use_ocr != "never" else None
        self.figure_extractor = FigureExtractor(ocr_extractor=self.ocr_extractor)
        self.pymupdf4llm_extractor = PyMuPDF4LLMExtractor(ocr_extractor=self.ocr_extractor)
        
        # Check GPU status for OCR
        gpu_available = False
        if self.ocr_extractor:
            from .extractors.ocr_extractor import check_gpu_available
            gpu_available = check_gpu_available()
        
        gpu_status = "GPU" if gpu_available else "CPU"
        logger.info(f"PyMuPDF4LLMProvider initialized: use_ocr={use_ocr}, lang={ocr_lang}, OCR={gpu_status}")
    
    def load(self, pdf_path: str | Path) -> PDFDocument:
        """
        Load PDF using PyMuPDF4LLM with markdown extraction
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PDFDocument object with markdown-formatted content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Loading PDF with PyMuPDF4LLM: {pdf_path.name}")
        
        # Open with PyMuPDF for metadata
        fitz_doc = fitz.open(str(pdf_path))
        total_pages = len(fitz_doc)
        
        # Extract metadata
        metadata = {
            "title": fitz_doc.metadata.get("title", ""),
            "author": fitz_doc.metadata.get("author", ""),
            "subject": fitz_doc.metadata.get("subject", ""),
            "creator": fitz_doc.metadata.get("creator", ""),
            "producer": fitz_doc.metadata.get("producer", ""),
            "page_count": total_pages,
        }
        
        logger.info(f"Total pages: {total_pages}")
        
        # Extract markdown using PyMuPDF4LLM (full document at once)
        markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
        
        # Split markdown by pages (PyMuPDF4LLM includes page markers)
        page_texts = self._split_markdown_by_pages(markdown_text, total_pages)
        
        # Process each page
        pages = []
        for page_num in range(total_pages):
            page_content = self._process_page(
                fitz_doc, 
                page_num, 
                page_texts[page_num] if page_num < len(page_texts) else ""
            )
            pages.append(page_content)
            
            logger.info(
                f"Page {page_num+1}/{total_pages}: {page_content.char_count} chars; "
                f"Figures: {len(page_content.figures)}; "
                f"Extraction: PyMuPDF4LLM"
            )
        
        fitz_doc.close()
        
        # Create document object
        doc = PDFDocument(
            file_path=str(pdf_path),
            total_pages=total_pages,
            pages=pages,
            metadata=metadata
        )
        
        logger.info(f"✅ Loaded {total_pages} pages from {pdf_path.name}")
        return doc
    
    def _split_markdown_by_pages(self, markdown_text: str, total_pages: int) -> list:
        """
        Split PyMuPDF4LLM markdown output by pages
        
        Args:
            markdown_text: Full markdown text from PyMuPDF4LLM
            total_pages: Total number of pages
            
        Returns:
            List of markdown text per page
        """
        # PyMuPDF4LLM uses "-----" as page separator
        page_texts = markdown_text.split("\n-----\n")
        
        # Ensure we have text for all pages
        while len(page_texts) < total_pages:
            page_texts.append("")
        
        return page_texts[:total_pages]
    
    def _process_page(self, 
                      fitz_doc, 
                      page_num: int,
                      markdown_text: str) -> PageContent:
        """
        Process single page with PyMuPDF4LLM markdown + OCR fallback
        
        Args:
            fitz_doc: PyMuPDF document
            page_num: Page number (0-indexed)
            markdown_text: Markdown text for this page
            
        Returns:
            PageContent with markdown text and extracted elements
        """
        page = fitz_doc[page_num]
        
        # Use extractor for unified processing
        result = self.pymupdf4llm_extractor.extract_page(fitz_doc, page, page_num)
        
        # Override text with PyMuPDF4LLM markdown if available
        if markdown_text and len(markdown_text.strip()) > 50:
            result['text'] = markdown_text
            result['extraction_method'] = 'pymupdf4llm'
        
        # Extract figures (group images + OCR)
        figures = self.figure_extractor.extract(fitz_doc, page, page_num)
        
        # Detect language
        language = self._detect_language(result['text'])
        
        return PageContent(
            page_number=page_num + 1,
            text=result['text'],
            tables=result['tables'],
            figures=figures,
            extraction_method=result['extraction_method'],
            char_count=len(result['text'].strip()),
            language=language
        )
    
    def _detect_language(self, text: str) -> str:
        """
        Auto-detect language per page
        
        Args:
            text: Page text content
            
        Returns:
            "en" for English, "zh" for Chinese, etc.
        """
        try:
            from pdf_extract_kit.utils.merge_blocks_and_spans import detect_lang
            return detect_lang(text) if text.strip() else "en"
        except ImportError:
            logger.debug("detect_lang not available, defaulting to 'en'")
            return "en"

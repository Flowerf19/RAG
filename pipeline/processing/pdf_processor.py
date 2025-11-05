"""
PDF Processor Module
====================
Handles PDF loading and chunking operations.
Single Responsibility: Convert PDF files to chunk sets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Processes PDF files into chunks using PDFProvider and SemanticChunker.
    """

    def __init__(self, loader: Any, chunker: Any, chunker_provider: Any):
        """
        Initialize PDF processor.
        
        Args:
            loader: PDFProvider instance
            chunker: SemanticChunker instance
            chunker_provider: SemanticChunkerProvider instance
        """
        self.loader = loader
        self.chunker = chunker
        self.chunker_provider = chunker_provider

    def process(self, pdf_path: Path) -> tuple[Any, Any]:
        """
        Load PDF and generate chunks.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (pdf_document, chunk_set)
        """
        logger.info(f"Processing PDF: {pdf_path.name}")

        # Step 1: Load PDF with smart extraction
        logger.info("Loading PDF with OCR + Layout detection...")
        pdf_doc = self.loader.load(str(pdf_path))

        logger.info(
            f"Loaded {pdf_doc.total_pages} pages - "
            f"Method: {[p.extraction_method for p in pdf_doc.pages[:3]]} - "
            f"Tables: {sum(len(p.tables) for p in pdf_doc.pages)} - "
            f"Figures: {sum(len(p.figures) for p in pdf_doc.pages)}"
        )

        # Step 2: Chunk document
        logger.info("Chunking document...")
        chunk_set = self.chunker.chunk(pdf_doc)
        logger.info(
            f"Created {len(chunk_set.chunks)} chunks, "
            f"strategy: {chunk_set.chunk_strategy}, "
            f"tokens: {chunk_set.total_tokens}"
        )

        return pdf_doc, chunk_set

    @staticmethod
    def validate_pdf_path(pdf_path: str | Path) -> Path:
        """
        Validate and resolve PDF file path.
        
        Args:
            pdf_path: Path to PDF file (str or Path)
            
        Returns:
            Resolved absolute Path object
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)

        # Resolve path properly
        if not pdf_path.is_absolute():
            # If relative path, make it absolute from current working directory
            pdf_path = pdf_path.resolve()

        # Verify file exists
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        return pdf_path


def create_pdf_processor(
    use_ocr: str = "auto",
    ocr_lang: str = "multilingual",
    min_text_threshold: int = 50,
) -> PDFProcessor:
    """
    Factory function to create PDFProcessor with default components.
    
    Args:
        use_ocr: OCR mode ("auto", "always", "never")
        ocr_lang: OCR language ("multilingual", "en", "vi", etc.)
        min_text_threshold: Minimum characters to consider text-based PDF
        
    Returns:
        Configured PDFProcessor instance
    """
    # Import dependencies
    from PDFLoaders.pdf_provider import PDFProvider
    from chunkers.semantic_chunker import SemanticChunker
    from chunkers.providers import create_semantic_provider

    # Initialize loader
    loader = PDFProvider(
        use_ocr=use_ocr,
        ocr_lang=ocr_lang,
        min_text_threshold=min_text_threshold,
    )
    logger.info("Using PDFProvider with smart OCR + Layout detection")

    # Initialize chunker
    chunker = SemanticChunker()

    # Initialize chunker provider
    chunker_provider = create_semantic_provider(
        normalize=True,
        stopwords=False,  # Keep stopwords for semantic meaning
        entities=True,  # Extract entities
        language=True,  # Detect language
        min_length=10,
    )
    logger.info("Using SemanticChunker with provider preprocessing")

    return PDFProcessor(loader, chunker, chunker_provider)

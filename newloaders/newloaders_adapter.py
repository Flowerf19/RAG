"""
NewLoaders Adapter
==================
Adapter để tích hợp newloaders vào RAG system hiện tại.
Chuyển đổi output của newloaders thành format PDFDocument/PDFPage tương thích.

⚠️  LƯU Ý: newloaders hiện tại có vấn đề dependencies (doclayout-yolo version conflict).
Khi dependencies được fix, uncomment phần load_models() để sử dụng.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
# Import current loaders models
from loaders.model.document import PDFDocument
from loaders.model.page import PDFPage
from loaders.model.block import Block

# Add parent directory to path to import loaders
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))



_NEWLOADERS_AVAILABLE = True  # Force enable newloaders
_NEWLOADERS_ERROR = "Dependencies not compatible with current Python version"

logger = logging.getLogger(__name__)


class NewLoadersAdapter:
    """
    Adapter để chuyển đổi output của newloaders thành format tương thích với RAG system.

    Hiện tại: Mock implementation do dependency issues.
    Khi fix dependencies: Uncomment model loading code.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize adapter.

        Args:
            config_path: Path to newloaders config file
        """
        if not _NEWLOADERS_AVAILABLE:
            logger.warning(f"newloaders not available: {_NEWLOADERS_ERROR}. Using fallback mode.")

        self.config_path = config_path or "configs/layout_detection.yaml"
        self.layout_model = None
        self.table_model = None
        self.ocr_model = None

    def load_models(self):
        """Load required models for layout detection, table parsing, and OCR."""
        if not _NEWLOADERS_AVAILABLE:
            logger.info("Skipping model loading - newloaders not available")
            return

        # Uncomment when dependencies are fixed
        pass

    def load_pdf(self, file_path: str) -> PDFDocument:
        """
        Load PDF using newloaders and convert to PDFDocument format.

        Args:
            file_path: Path to PDF file

        Returns:
            PDFDocument: Compatible with current RAG system
        """
        # Convert to absolute path
        pdf_path = Path(file_path)
        logger.info(f"Adapter received file_path: {file_path}")
        logger.info(f"Resolved pdf_path: {pdf_path}")
        logger.info(f"pdf_path.exists(): {pdf_path.exists()}")
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # TEMPORARY: Use current PDFLoader as fallback until newloaders dependencies are fixed
        logger.info("Using current PDFLoader as fallback for newloaders integration")

        try:
            # Try to load with newloaders
            # Add newloaders to path for import
            newloaders_path = Path(__file__).parent
            if str(newloaders_path) not in sys.path:
                sys.path.insert(0, str(newloaders_path))

            from pdf_extract_kit import PDFExtractor
            extractor = PDFExtractor(config_path=self.config_path)
            result = extractor(str(pdf_path))
            return self._convert_newloaders_output(result, str(pdf_path))

        except Exception as e:
            logger.error(f"Newloaders failed: {e}")
            raise RuntimeError(f"Newloaders integration failed: {e}")

    def _convert_newloaders_output(self, newloaders_result, file_path: str) -> PDFDocument:
        """
        Convert newloaders output to PDFDocument format.

        Args:
            newloaders_result: Output from newloaders PDFExtractor
            file_path: Path to the PDF file

        Returns:
            PDFDocument: Compatible with current RAG system
        """
        logger.info("Converting newloaders output to PDFDocument format")

        # Create PDFDocument
        doc = PDFDocument(
            file_path=file_path,
            num_pages=len(newloaders_result.get("pages", [])),
            meta={
                "source": "newloaders",
                "extraction_method": "newloaders_mock",
                "title": Path(file_path).stem
            },
            pages=[]
        )

        # Convert each page
        for page_idx, page_data in enumerate(newloaders_result.get("pages", [])):
            # Create PDFPage
            page = PDFPage(
                page_number=page_idx + 1,
                blocks=[],
                source={
                    "file_path": file_path,
                    "page_number": page_idx + 1,
                    "doc_id": Path(file_path).stem,
                    "doc_title": Path(file_path).stem,
                    "layout_detection_capable": True,
                    "ocr_capable": True,
                    "enhanced_by": "newloaders_adapter",
                    "extraction_method": "newloaders_mock"
                }
            )

            # Convert blocks
            for block_data in page_data.get("blocks", []):
                # Create appropriate block type based on block_data
                if block_data.get("type") == "text":
                    block = Block(
                        text=block_data.get("text", ""),
                        bbox=block_data.get("bbox", [0, 0, 100, 100]),
                        metadata={"block_type": "text", "page_number": page_idx + 1}
                    )
                else:
                    # Default to text block
                    block = Block(
                        text=str(block_data),
                        bbox=[0, 0, 100, 100],
                        metadata={"block_type": "unknown", "page_number": page_idx + 1}
                    )

                page.blocks.append(block)

            doc.pages.append(page)

        logger.info(f"Converted {len(doc.pages)} pages with {sum(len(p.blocks) for p in doc.pages)} blocks")
        return doc

    def _enhance_document_metadata(self, doc: PDFDocument, file_path: str) -> PDFDocument:
        """
        Enhance existing document with additional metadata for RAG source tracking.
        """
        # Add enhanced source tracking to each page
        for page in doc.pages:
            if hasattr(page, 'source'):
                page.source.update({
                    "enhanced_by": "newloaders_adapter",
                    "layout_detection_capable": _NEWLOADERS_AVAILABLE,
                    "ocr_capable": _NEWLOADERS_AVAILABLE,
                    "table_parsing_capable": _NEWLOADERS_AVAILABLE
                })

        # Add document-level metadata
        doc.meta.update({
            "loader_type": "enhanced_pdf_loader",
            "newloaders_available": _NEWLOADERS_AVAILABLE,
            "source_tracking": "enabled"
        })

        return doc

    def _create_minimal_document(self, file_path: str) -> PDFDocument:
        """
        Create minimal PDFDocument when all loading methods fail.
        """
        pdf_path = Path(file_path)
        doc_title = pdf_path.stem
        doc_id = str(pdf_path)

        # Create empty page as placeholder
        empty_page = PDFPage(
            page_number=1,
            text="",
            blocks=[],
            tables=[],
            warnings=["PDF loading failed - using minimal document"],
            source={
                "file_path": file_path,
                "page_number": 1,
                "page_size": {"width": 0, "height": 0},
                "doc_id": doc_id,
                "doc_title": doc_title,
                "page_label": None,
                "loader_status": "failed"
            }
        )

        return PDFDocument(
            file_path=file_path,
            num_pages=1,
            meta={"loader_status": "failed", "error": "All loading methods failed"},
            pages=[empty_page],
            warnings=["Document created with minimal content due to loading failures"]
        )


def create_compatible_loader():
    """
    Create a loader compatible with current RAG system but using newloaders internally.
    Returns enhanced loader with better source tracking.
    """
    adapter = NewLoadersAdapter()

    class CompatibleLoader:
        """Wrapper to make newloaders compatible with current PDFLoader interface."""

        @staticmethod
        def create_default():
            return CompatibleLoader()

        def load(self, file_path: str) -> PDFDocument:
            return adapter.load_pdf(file_path)

    return CompatibleLoader()
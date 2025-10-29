import os
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF for PDF processing

from pdf_extract_kit.tasks import LayoutDetectionTask, OCRTask, TableParsingTask
from pdf_extract_kit.registry.registry import MODEL_REGISTRY

current_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.abspath(os.path.join(current_dir, '..'))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


class PDFExtractor:
    def __init__(self, config_path=None):
        """
        Initialize PDFExtractor with layout detection, OCR, and table parsing capabilities.

        Args:
            config_path: Path to config file (optional)
        """
        self.config_path = config_path or "configs/layout_detection.yaml"

        # Initialize models
        try:
            # Try to load layout detection model
            layout_model = MODEL_REGISTRY.get("doclayout_yolo")()
            self.layout_detector = LayoutDetectionTask(layout_model)
        except Exception as e:
            print(f"Warning: Could not load layout detection model: {e}")
            self.layout_detector = None

        try:
            # Try to load OCR model
            ocr_model = MODEL_REGISTRY.get("paddleocr")()
            self.ocr_detector = OCRTask(ocr_model)
        except Exception as e:
            print(f"Warning: Could not load OCR model: {e}")
            self.ocr_detector = None

        try:
            # Try to load table parsing model
            table_model = MODEL_REGISTRY.get("struct_eqtable")()
            self.table_parser = TableParsingTask(table_model)
        except Exception as e:
            print(f"Warning: Could not load table parsing model: {e}")
            self.table_parser = None

    def __call__(self, pdf_path):
        """
        Extract content from PDF using available models.

        Args:
            pdf_path: Path to PDF file

        Returns:
            dict: Extracted content with pages, layout, text, tables
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Open PDF with PyMuPDF
        doc = fitz.open(str(pdf_path))
        pages_data = []

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]

                # Extract basic text and layout
                text = page.get_text()

                # Get page dimensions
                page_rect = page.rect
                width, height = page_rect.width, page_rect.height

                # Create page data structure
                page_data = {
                    "page_number": page_idx + 1,
                    "width": width,
                    "height": height,
                    "text": text,
                    "blocks": [],
                    "tables": [],
                    "layout_regions": []
                }

                # Try layout detection if available
                if self.layout_detector:
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Convert page to image
                            pix = page.get_pixmap(dpi=150)
                            img_path = os.path.join(temp_dir, f"page_{page_idx + 1}.png")
                            pix.save(img_path)

                            # Detect layout
                            layout_results = self.layout_detector.predict_images(img_path, temp_dir)
                            page_data["layout_regions"] = layout_results
                    except Exception as e:
                        print(f"Warning: Layout detection failed for page {page_idx + 1}: {e}")

                # Try OCR if available and text is minimal
                if self.ocr_detector and len(text.strip()) < 100:
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Convert page to image
                            pix = page.get_pixmap(dpi=300)
                            img_path = os.path.join(temp_dir, f"page_{page_idx + 1}.png")
                            pix.save(img_path)

                            # Perform OCR
                            ocr_results = self.ocr_detector.predict_images(img_path, temp_dir)
                            if ocr_results:
                                # Add OCR text to page data
                                ocr_text = " ".join([result.get("text", "") for result in ocr_results])
                                if ocr_text:
                                    page_data["ocr_text"] = ocr_text
                                    if not page_data["text"]:
                                        page_data["text"] = ocr_text
                    except Exception as e:
                        print(f"Warning: OCR failed for page {page_idx + 1}: {e}")

                # Try table extraction if available
                if self.table_parser:
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Convert page to image
                            pix = page.get_pixmap(dpi=150)
                            img_path = os.path.join(temp_dir, f"page_{page_idx + 1}.png")
                            pix.save(img_path)

                            # Detect tables
                            table_results = self.table_parser.predict_images(img_path, temp_dir)
                            page_data["tables"] = table_results
                    except Exception as e:
                        print(f"Warning: Table parsing failed for page {page_idx + 1}: {e}")

                # Create blocks from text and layout
                page_data["blocks"] = self._create_blocks_from_text(text, page_data)

                pages_data.append(page_data)

        finally:
            doc.close()

        return {
            "pages": pages_data,
            "metadata": {
                "source": "newloaders_real",
                "total_pages": len(pages_data),
                "capabilities": {
                    "layout_detection": self.layout_detector is not None,
                    "ocr": self.ocr_detector is not None,
                    "table_parsing": self.table_parser is not None
                }
            }
        }

    def _create_blocks_from_text(self, text, page_data):
        """
        Create text blocks from extracted text and layout information.
        """
        blocks = []

        # Simple block creation - split by paragraphs
        paragraphs = text.split('\n\n')
        y_offset = 0

        for para in paragraphs:
            if para.strip():
                block = {
                    "type": "text",
                    "text": para.strip(),
                    "bbox": [50, y_offset, 500, y_offset + 50],  # Placeholder bbox
                    "confidence": 0.9
                }
                blocks.append(block)
                y_offset += 60

        return blocks


__all__ = ['PDFExtractor']
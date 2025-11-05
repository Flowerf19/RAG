"""
PDF Provider - Smart PDF loading with OCR integration
"""

import fitz
from pathlib import Path
import logging
import warnings

from .models import PDFDocument, PageContent
from .extractors import OCRExtractor, TableExtractor, FigureExtractor

# Suppress pdfminer.six warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pdfminer')
logging.getLogger('pdfminer').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class PDFProvider:
    """
    Smart PDF Provider - Tự động quyết định extraction strategy
    
    Logic:
    - Text-based PDF (>100 chars/page) → PyMuPDF text extraction
    - Image-based PDF (<50 chars/page) → PaddleOCR
    - Mixed PDF → Hybrid per page
    - Tables → pdfplumber + OCR enhancement
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
        self.table_extractor = TableExtractor(ocr_extractor=self.ocr_extractor)
        self.figure_extractor = FigureExtractor(ocr_extractor=self.ocr_extractor)
        
        # Check GPU status for OCR
        gpu_available = False
        if self.ocr_extractor:
            from .extractors.ocr_extractor import check_gpu_available
            gpu_available = check_gpu_available()
        
        # Check PDF-Extract-Kit task availability
        kit_tasks_status = self._check_kit_tasks_availability()
        
        gpu_status = "GPU" if gpu_available else "CPU"
        logger.info(f"PDFProvider initialized: use_ocr={use_ocr}, lang={ocr_lang}, OCR={gpu_status}")
        if kit_tasks_status:
            logger.info(f"PDF-Extract-Kit tasks available: {kit_tasks_status}")
    
    def _check_kit_tasks_availability(self) -> str:
        """Check which PDF-Extract-Kit tasks are available"""
        try:
            from PDFLoaders.pdf_extract_kit.tasks import (
                _layout_available, _formula_available, _formula_recog_available,
                _ocr_available, _table_available
            )
            available_tasks = []
            if _layout_available:
                available_tasks.append("Layout")
            if _formula_available:
                available_tasks.append("Formula")
            if _formula_recog_available:
                available_tasks.append("FormulaRecog")
            if _ocr_available:
                available_tasks.append("OCR")
            if _table_available:
                available_tasks.append("Table")
            
            if available_tasks:
                return f"{len(available_tasks)}/5 ({', '.join(available_tasks)})"
            return ""
        except ImportError:
            return ""
    
    def load(self, pdf_path: str | Path) -> PDFDocument:
        """
        Load PDF với smart extraction strategy
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PDFDocument object với full content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Loading PDF: {pdf_path.name}")
        
        # Open with PyMuPDF for metadata and basic info
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
        
        # Process each page
        pages = []
        for page_num in range(total_pages):
            page_content = self._process_page(
                fitz_doc, 
                page_num, 
                pdf_path
            )
            pages.append(page_content)
            
            # Compute enhanced table / figure stats for clearer logging
            tables_count = len(page_content.tables)
            tables_enhanced = 0
            for t in page_content.tables:
                try:
                    # TableExtractor adds an OCR supplement row when enhanced
                    if isinstance(t, list) and t and any(isinstance(r, list) and len(r) > 0 and isinstance(r[0], str) and r[0].startswith("[OCR Supplement]") for r in [t[-1]]):
                        tables_enhanced += 1
                except Exception:
                    continue

            figures_count = len(page_content.figures)
            figures_with_text = sum(1 for f in page_content.figures if f.get("text"))

            # Human-friendly method labels
            text_method_label = "PaddleOCR" if page_content.extraction_method == "ocr" else "PyMuPDF(text)"
            tables_method_label = "pdfplumber" + ("+OCR" if tables_enhanced > 0 else "")
            figures_method_label = "OCR(text extracted)" if figures_with_text > 0 else "images-only"

            logger.info(
                f"Page {page_num+1}/{total_pages}: {page_content.char_count} chars; "
                f"Tables: {tables_count} [{tables_method_label}]; "
                f"Figures: {figures_count} [{figures_method_label}]; "
                f"Page-extraction: {text_method_label}"
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
    
    def _process_page(self, 
                      fitz_doc, 
                      page_num: int,
                      pdf_path: Path) -> PageContent:
        """
        Process single page với smart strategy
        
        Args:
            fitz_doc: PyMuPDF document
            page_num: Page number (0-indexed)
            pdf_path: PDF file path
            
        Returns:
            PageContent with all extracted data
        """
        page = fitz_doc[page_num]
        
        # Step 1: Try text extraction first
        text = page.get_text()
        char_count = len(text.strip())
        
        # Step 2: Decide extraction method
        if self.use_ocr == "always":
            # Force OCR
            extraction_method = "ocr"
            text = self.ocr_extractor.extract(page, page_num)
            char_count = len(text.strip())
        elif self.use_ocr == "never":
            # Text only
            extraction_method = "text"
        else:
            # Auto detect
            if char_count < self.min_text_threshold:
                # Image-based page → need OCR
                extraction_method = "ocr"
                text = self.ocr_extractor.extract(page, page_num)
                char_count = len(text.strip())
            else:
                # Text-based page
                extraction_method = "text"
        
        # Step 3: Extract tables (with optional OCR enhancement)
        tables = self.table_extractor.extract(page_num, pdf_path, fitz_page=page)
        
        # Step 4: Extract figures (group images + OCR)
        figures = self.figure_extractor.extract(fitz_doc, page, page_num)
        
        # Step 5: Detect language
        language = self._detect_language(text)
        
        return PageContent(
            page_number=page_num + 1,
            text=text,
            tables=tables,
            figures=figures,
            extraction_method=extraction_method,
            char_count=char_count,
            language=language
        )
    
    def _detect_language(self, text: str) -> str:
        """
        Auto-detect language per page using PDF-Extract-Kit utility
        
        Args:
            text: Page text content
            
        Returns:
            "en" for English, "zh" for Chinese
        """
        try:
            from pdf_extract_kit.utils.merge_blocks_and_spans import detect_lang
            return detect_lang(text) if text.strip() else "en"
        except ImportError:
            logger.debug("detect_lang not available, defaulting to 'en'")
            return "en"
    
    def detect_pdf_type(self, pdf_path: str | Path) -> str:
        """
        Detect PDF type: "text", "image", "mixed"
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            "text": text-based PDF (>100 chars/page avg)
            "image": image-based PDF (<50 chars/page avg)
            "mixed": mixed content
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        
        # Sample first 3 pages
        sample_size = min(3, len(doc))
        total_chars = 0
        
        for page_num in range(sample_size):
            page = doc[page_num]
            text = page.get_text()
            total_chars += len(text.strip())
        
        doc.close()
        
        avg_chars = total_chars / sample_size if sample_size > 0 else 0
        
        if avg_chars < 50:
            return "image"
        elif avg_chars > 100:
            return "text"
        else:
            return "mixed"
    
    def export_to_markdown(self, 
                          doc: PDFDocument, 
                          output_path: str | Path) -> Path:
        """
        Export document to markdown format
        
        Args:
            doc: PDFDocument to export
            output_path: Output markdown file path
            
        Returns:
            Path to saved markdown file
        """
        from pdf_extract_kit.utils.merge_blocks_and_spans import ocr_escape_special_markdown_char
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        lines.append(f"# {doc.metadata.get('title', 'Untitled Document')}\n")
        lines.append(f"**Author:** {doc.metadata.get('author', 'Unknown')}\n")
        lines.append(f"**Pages:** {doc.total_pages}\n")
        lines.append("\n---\n\n")
        
        for page in doc.pages:
            lines.append(f"## Page {page.page_number}\n\n")
            
            # Add text content
            if page.text.strip():
                escaped_text = ocr_escape_special_markdown_char(page.text.strip())
                lines.append(f"{escaped_text}\n\n")
            
            # Add tables
            for i, table in enumerate(page.tables, 1):
                lines.append(f"### Table {page.page_number}.{i}\n\n")
                
                if table:
                    # Header row
                    header = table[0]
                    lines.append("| " + " | ".join(str(cell or "") for cell in header) + " |\n")
                    lines.append("| " + " | ".join("---" for _ in header) + " |\n")
                    
                    # Data rows
                    for row in table[1:]:
                        lines.append("| " + " | ".join(str(cell or "") for cell in row) + " |\n")
                    
                    lines.append("\n")
            
            # Add figure placeholders
            for i, fig in enumerate(page.figures, 1):
                lines.append(f"### Figure {page.page_number}.{i}\n\n")
                lines.append(f"**Type:** {fig['type']}\n")
                lines.append(f"**Images:** {fig['image_count']}\n")
                
                # OCR text from figure
                if fig.get('text'):
                    lines.append(f"\n**Text:** {fig['text']}\n")
                lines.append("\n")
            
            lines.append("\n---\n\n")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        logger.info(f"✅ Markdown exported: {output_path}")
        return output_path
    
    def export_figure_images(self, 
                            pdf_path: str | Path,
                            doc: PDFDocument,
                            output_dir: str = "data/figures") -> PDFDocument:
        """
        Extract and save actual image data for each figure
        
        Args:
            pdf_path: Path to source PDF
            doc: PDFDocument with figures
            output_dir: Base output directory
            
        Returns:
            Updated PDFDocument with image_path in figures
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir) / pdf_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fitz_doc = fitz.open(str(pdf_path))
        
        for page_content in doc.pages:
            
            for fig_idx, figure in enumerate(page_content.figures, 1):
                try:
                    # Get first image from figure
                    if figure['images']:
                        img_info = figure['images'][0]
                        xref = img_info['xref']
                        
                        # Extract image
                        base_image = fitz_doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Save image
                        img_filename = f"page_{page_content.page_number}_fig_{fig_idx}.png"
                        img_path = output_dir / img_filename
                        
                        with open(img_path, 'wb') as f:
                            f.write(image_bytes)
                        
                        # Update figure with image path
                        figure['image_path'] = str(img_path)
                        
                        logger.debug(f"Saved figure image: {img_path}")
                        
                except Exception as e:
                    logger.warning(f"Failed to extract image for page {page_content.page_number} fig {fig_idx}: {e}")
        
        fitz_doc.close()
        logger.info(f"✅ Figure images exported to: {output_dir}")
        
        return doc

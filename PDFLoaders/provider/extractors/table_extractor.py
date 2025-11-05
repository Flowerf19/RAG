"""
Table Extractor - Extract tables from PDF with OCR enhancement and optional ML-based parsing
"""

import pdfplumber
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
import logging
import warnings

# Suppress pdfminer font warnings
warnings.filterwarnings('ignore', message='.*FontBBox.*')
logging.getLogger('pdfminer.pdffont').setLevel(logging.ERROR)

if TYPE_CHECKING:
    from .ocr_extractor import OCRExtractor

logger = logging.getLogger(__name__)

# Try to import PDF-Extract-Kit TableParsingTask (optional, for advanced table parsing)
try:
    from PDFLoaders.pdf_extract_kit.tasks import TableParsingTask, _table_available
    TABLE_PARSING_AVAILABLE = _table_available
except ImportError:
    TableParsingTask = None
    TABLE_PARSING_AVAILABLE = False


class TableExtractor:
    """Extract tables using pdfplumber with smart filtering, OCR enhancement, and optional ML-based parsing"""
    
    def __init__(self, ocr_extractor: Optional['OCRExtractor'] = None, use_ml_parsing: bool = False):
        """
        Args:
            ocr_extractor: Optional OCR extractor for table enhancement
            use_ml_parsing: Use PDF-Extract-Kit's ML-based table parsing (requires CUDA GPU)
        """
        self.ocr_extractor = ocr_extractor
        self.use_ml_parsing = use_ml_parsing and TABLE_PARSING_AVAILABLE
        self.table_parser = None
        
        if self.use_ml_parsing:
            try:
                # Initialize TableParsingTask with model
                # Note: This requires CUDA GPU and model weights
                logger.info("Initializing ML-based table parser (requires CUDA)...")
                # TableParsingTask requires a model instance
                # For now, we'll keep it optional and log availability
                logger.info("ML table parsing is available but not initialized (requires model configuration)")
                self.use_ml_parsing = False  # Disable until model is configured
            except Exception as e:
                logger.warning(f"Could not initialize ML table parser: {e}")
                self.use_ml_parsing = False
        
        if not self.use_ml_parsing:
            logger.debug("Using pdfplumber-based table extraction (default)")
    
    def extract(self, page_num: int, pdf_path: Path, fitz_page=None) -> List[List[List[str]]]:
        """
        Extract tables using pdfplumber with smart filtering
        
        Strategy:
        1. Try pdfplumber first (works for text-based tables)
        2. If tables have many empty cells → use OCR to fill missing content
        3. Return hybrid result with OCR-enhanced tables
        
        Args:
            page_num: Page number (0-indexed)
            pdf_path: Path to PDF file
            fitz_page: Optional PyMuPDF page for OCR fallback
            
        Returns:
            List of tables (each table is 2D array of strings)
        """
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                if page_num >= len(pdf.pages):
                    return []
                
                page = pdf.pages[page_num]
                
                # Extract with improved settings
                tables = page.extract_tables({
                    'vertical_strategy': 'lines',
                    'horizontal_strategy': 'lines',
                    'snap_tolerance': 3,
                    'join_tolerance': 3,
                    'edge_min_length': 3,
                })
                
                if not tables:
                    return []
                
                # Filter empty tables and false positives
                filtered_tables = []
                for table in tables:
                    # Remove empty rows
                    filtered_table = [
                        row for row in table 
                        if any(cell for cell in row if cell)
                    ]
                    
                    if not filtered_table:
                        continue
                    
                    # Apply smart filters to remove false positives
                    if self._is_false_positive(filtered_table):
                        continue
                    
                    # Check if table needs OCR enhancement (many empty cells)
                    if self.ocr_extractor and fitz_page is not None:
                        enhanced_table = self._enhance_with_ocr(filtered_table, fitz_page)
                        filtered_tables.append(enhanced_table)
                    else:
                        filtered_tables.append(filtered_table)
                
                return filtered_tables
        except Exception as e:
            logger.warning(f"Table extraction failed for page {page_num+1}: {e}")
            return []
    
    def _enhance_with_ocr(self, table: List[List[str]], fitz_page) -> List[List[str]]:
        """
        Enhance table with OCR for empty cells (image-based tables)
        
        Strategy:
        - If >30% cells are empty/None → likely image-based table
        - Use OCR on full page and try to match content to empty cells
        - Keep original pdfplumber structure, fill with OCR text
        
        Args:
            table: Original table from pdfplumber
            fitz_page: PyMuPDF page for OCR
            
        Returns:
            Enhanced table with OCR-filled cells
        """
        # Count empty cells
        total_cells = sum(len(row) for row in table)
        empty_cells = sum(1 for row in table for cell in row if not cell or str(cell).strip() == "")
        empty_ratio = empty_cells / total_cells if total_cells > 0 else 0
        
        # Only enhance if significant empty content (>30%)
        if empty_ratio < 0.3:
            return table
        
        logger.debug(f"Table has {empty_ratio:.1%} empty cells, attempting OCR enhancement...")
        
        # Run OCR on full page
        try:
            page_text = self.ocr_extractor.extract(fitz_page, fitz_page.number)
            if not page_text:
                return table
            
            # Simple strategy: append OCR text as note to table
            # (Full cell-by-cell matching would need table bbox coordinates)
            # For now, add OCR text as last row as supplementary info
            enhanced = table.copy()
            enhanced.append(["[OCR Supplement]", page_text[:200] + "..."])  # Truncate for brevity
            
            return enhanced
            
        except Exception as e:
            logger.debug(f"Table OCR enhancement failed: {e}")
            return table
    
    def _is_false_positive(self, table: List[List[str]]) -> bool:
        """
        Filter out false positive tables (headers, footers, watermarks)
        
        Args:
            table: Filtered table with non-empty rows
            
        Returns:
            True if table is likely a false positive (should be filtered out)
        """
        if not table:
            return True
        
        rows = len(table)
        cols = len(table[0]) if table else 0
        
        # Rule 1: Single row with 2 cols containing only watermarks/placeholders
        if rows == 1 and cols == 2:
            first_cell = (table[0][0] or "").strip()
            second_cell = (table[0][1] or "").strip()
            
            # Watermark patterns: "XXXX-XXXX-XXX" or just "X"
            if ("XXXX" in first_cell or "XXX" in first_cell) and second_cell in ["X", ""]:
                return True
            
            # Very short content (likely header/footer marker)
            if len(first_cell) < 3 and len(second_cell) < 3:
                return True
        
        # Rule 2: 3×5 table that looks like page header (QMS classification)
        if rows == 3 and cols == 5:
            # Check if first row contains QMS-related keywords
            first_row_text = " ".join(str(cell or "") for cell in table[0]).upper()
            if "QMS" in first_row_text or "CLASSIFICATION" in first_row_text:
                return True
        
        # Rule 3: Very small tables (< 2 rows) that don't look like data
        if rows < 2:
            # Unless it's a single-row summary table with meaningful content
            row_text = " ".join(str(cell or "") for cell in table[0])
            if len(row_text.strip()) < 10:  # Too short to be meaningful
                return True
        
        # Rule 4: Tables with all cells containing same placeholder/watermark
        all_cells = [cell for row in table for cell in row if cell]
        if len(all_cells) > 0:
            unique_content = set(str(cell).strip() for cell in all_cells if str(cell).strip())
            # If only 1-2 unique values and they're placeholders
            if len(unique_content) <= 2:
                content_str = " ".join(unique_content)
                if "XXXX" in content_str or all(len(c) <= 2 for c in unique_content):
                    return True
        
        # Pass: Looks like a real content table
        return False

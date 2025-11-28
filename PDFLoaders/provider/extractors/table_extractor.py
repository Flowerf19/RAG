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
                
                # Extract with improved settings and bbox
                table_settings = {
                    'vertical_strategy': 'lines',
                    'horizontal_strategy': 'lines',
                    'snap_tolerance': 3,
                    'join_tolerance': 3,
                    'edge_min_length': 3,
                }
                
                tables = page.extract_tables(table_settings)
                
                # Also get table objects with bbox
                table_objects = page.find_tables(table_settings=table_settings)
                
                if not tables:
                    return []
                
                # Filter empty tables and false positives
                filtered_tables = []
                for i, table in enumerate(tables):
                    # Remove empty rows
                    filtered_table = [
                        row for row in table 
                        if any(cell for cell in row if cell)
                    ]
                    
                    if not filtered_table:
                        continue
                    
                    # Get bbox for this table
                    bbox = None
                    if i < len(table_objects):
                        bbox = table_objects[i].bbox
                    
                    # Apply smart filters to remove false positives
                    if self._is_false_positive(filtered_table, bbox):
                        continue
                    
                    # Check if table needs OCR enhancement (many empty cells)
                    if self.ocr_extractor and fitz_page is not None:
                        enhanced_table = self._enhance_with_ocr(filtered_table, fitz_page, page_num)
                        filtered_tables.append(enhanced_table)
                    else:
                        filtered_tables.append(filtered_table)
                
                return filtered_tables
        except Exception as e:
            logger.warning(f"Table extraction failed for page {page_num+1}: {e}")
            return []
    
    def _enhance_with_ocr(self, table: List[List[str]], fitz_page, page_num: int) -> List[List[str]]:
        """
        Enhance table with OCR for empty cells (image-based tables)
        
        Strategy:
        - If >30% cells are empty/None → likely image-based table
        - Use OCR on full page and try to match content to empty cells
        - Keep original pdfplumber structure, fill with OCR text
        
        Args:
            table: Original table from pdfplumber
            fitz_page: PyMuPDF page for OCR
            page_num: Page number for OCR processing
            
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
        
        # Use OCR to fill empty cells
        try:
            # Get OCR text for the entire page
            ocr_text = self.ocr_extractor.extract(fitz_page, page_num)
            
            # Simple strategy: if table is mostly empty, replace with OCR content
            # For now, append OCR text as a new row to indicate enhancement
            if empty_ratio > 0.9:  # Completely empty table
                # Create a new table from OCR text (simplified)
                ocr_lines = ocr_text.split('\n')[:len(table)]  # Limit to table size
                enhanced_table = []
                for i, row in enumerate(table):
                    if i < len(ocr_lines):
                        # Replace empty row with OCR line split into cells
                        ocr_cells = ocr_lines[i].split()[:len(row)]
                        enhanced_row = [cell if cell else ocr_cells[j] if j < len(ocr_cells) else "" 
                                       for j, cell in enumerate(row)]
                        enhanced_table.append(enhanced_row)
                    else:
                        enhanced_table.append(row)
                return enhanced_table
            else:
                # Fill individual empty cells with OCR text segments
                ocr_words = ocr_text.split()
                word_idx = 0
                enhanced_table = []
                for row in table:
                    enhanced_row = []
                    for cell in row:
                        if not cell or str(cell).strip() == "":
                            if word_idx < len(ocr_words):
                                enhanced_row.append(ocr_words[word_idx])
                                word_idx += 1
                            else:
                                enhanced_row.append("")
                        else:
                            enhanced_row.append(cell)
                    enhanced_table.append(enhanced_row)
                return enhanced_table
                
        except Exception as e:
            logger.warning(f"OCR enhancement failed: {e}")
            return table
    
    def _is_false_positive(self, table: List[List[str]], bbox: tuple = None) -> bool:
        """
        Enhanced filtering to remove false positive tables (headers, footers, watermarks)

        Rules:
        1. Header tables (3x3 with classification info)
        2. Single row with watermarks/placeholders
        3. Very small tables with no meaningful content
        4. Tables with all placeholder content
        5. Tables with repeated header-like content
        6. Header/footer tables based on position (bbox)
        """
        if not table:
            return True

        rows = len(table)
        cols = len(table[0]) if table else 0

        # Rule 6: Header/Footer tables based on bbox position
        if bbox:
            x0, y0, x1, y1 = bbox
            table_height = y1 - y0
            
            # Header tables: top 15% of page, small height (< 50 points)
            if y0 < 100 and table_height < 50:
                # Check for header content patterns
                all_text = " ".join(str(cell or "") for row in table for cell in row).upper()
                header_keywords = ["CLASSIFICATION", "OWNER", "COMPANY", "QMS", "VERSION", "INTERNAL"]
                if any(keyword in all_text for keyword in header_keywords):
                    return True
            
            # Footer tables: bottom 10% of page, very small
            if y0 > 700 and table_height < 20:  # Assuming A4 page height ~800-900 points
                return True

        # Rule 1: Document header tables (3x3 with classification/owner info)
        if rows == 3 and cols == 3:
            # Check for header patterns
            all_text = " ".join(str(cell or "") for row in table for cell in row).upper()
            header_keywords = ["CLASSIFICATION", "OWNER", "INFORMATION SECURITY", "ISMS/PR_", "VERSION"]
            if any(keyword in all_text for keyword in header_keywords):
                return True

        # Rule 2: Single row with 2 cols containing only watermarks/placeholders
        if rows == 1 and cols == 2:
            first_cell = (table[0][0] or "").strip()
            second_cell = (table[0][1] or "").strip()

            # Watermark patterns: "XXXX-XXXX-XXX" or just "X"
            if ("XXXX" in first_cell or "XXX" in first_cell) and second_cell in ["X", ""]:
                return True

            # Very short content (likely header/footer marker)
            if len(first_cell) < 3 and len(second_cell) < 3:
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

        # Rule 5: Tables that are just repeated headers or footers
        if rows >= 2:
            # Check if all rows are very similar (repeated headers)
            row_texts = []
            for row in table:
                row_text = " ".join(str(cell or "").strip() for cell in row)
                row_texts.append(row_text)

            # If most rows are identical or very similar
            unique_rows = set(row_texts)
            if len(unique_rows) <= rows * 0.3:  # Less than 30% unique rows
                return True

        # Pass: Looks like a real content table
        return False

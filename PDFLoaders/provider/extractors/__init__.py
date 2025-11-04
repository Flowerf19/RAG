"""
Extractors package - Text, OCR, Table, Figure extraction
"""

from .ocr_extractor import OCRExtractor
from .table_extractor import TableExtractor
from .figure_extractor import FigureExtractor

__all__ = ['OCRExtractor', 'TableExtractor', 'FigureExtractor']

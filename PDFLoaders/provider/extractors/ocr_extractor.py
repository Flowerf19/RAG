"""
OCR Extractor - PaddleOCR integration for text extraction
"""

import fitz
from pathlib import Path
import logging
from typing import Optional, Any
import os

# Optional PaddleOCR - graceful degradation
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

logger = logging.getLogger(__name__)


class OCRExtractor:
    """Extract text using PaddleOCR"""
    
    # PaddleOCR language mapping
    LANG_MAP = {
        # Multilingual & English
        "multilingual": "en",
        "en": "en",
        "english": "en",
        
        # Asian languages
        "ch": "ch",
        "chinese": "ch",
        "zh": "ch",
        "korean": "korean",
        "ko": "korean",
        "japan": "japan",
        "japanese": "japan",
        "ja": "japan",
        
        # European languages (Latin-based)
        "vi": "latin",
        "vietnamese": "latin",
        "fr": "latin",  # French
        "french": "latin",
        "de": "latin",  # German
        "german": "latin",
        "es": "latin",  # Spanish
        "spanish": "latin",
        "it": "latin",  # Italian
        "italian": "latin",
        "pt": "latin",  # Portuguese
        "portuguese": "latin",
        "pl": "latin",  # Polish
        "polish": "latin",
        "nl": "latin",  # Dutch
        "dutch": "latin",
        
        # Cyrillic script (Russian, etc.)
        "ru": "cyrillic",
        "russian": "cyrillic",
        "uk": "cyrillic",  # Ukrainian
        "ukrainian": "cyrillic",
        
        # Arabic script
        "ar": "arabic",
        "arabic": "arabic",
        
        # Other scripts
        "devanagari": "devanagari",  # Hindi, Sanskrit, etc.
        "hi": "devanagari",
        "hindi": "devanagari",
    }
    
    def __init__(self, lang: str = "multilingual"):
        """
        Args:
            lang: Language for OCR ("multilingual", "en", "ch", "vi", etc.)
                  Will be mapped to PaddleOCR supported languages
        """
        self.input_lang = lang
        self.paddle_lang = self.LANG_MAP.get(lang.lower(), "en")
        self.ocr_engine: Optional[Any] = None
    
    def _init_ocr(self):
        """Initialize OCR engine if not already initialized"""
        if self.ocr_engine is None and PADDLEOCR_AVAILABLE:
            logger.info(f"Initializing PaddleOCR (input_lang={self.input_lang} → paddle_lang={self.paddle_lang})...")
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=self.paddle_lang,  # Use mapped language
                show_log=False,
                use_gpu=False
            )
    
    def extract(self, page, page_num: int) -> str:
        """
        Extract text using PaddleOCR
        
        Args:
            page: PyMuPDF page object
            page_num: Page number for temp file naming
            
        Returns:
            Extracted text
        """
        if not PADDLEOCR_AVAILABLE:
            logger.warning("PaddleOCR not available, falling back to PyMuPDF text")
            return page.get_text()
        
        # Initialize OCR engine if needed
        self._init_ocr()
        
        # Convert page to image and save to cache directory
        cache_dir = Path("data/cache/ocr_temp")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 2x scale for better OCR accuracy
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        
        # Save with unique name based on page number
        temp_img_path = cache_dir / f"page_{page_num}_temp.png"
        pix.save(str(temp_img_path))
        
        try:
            # Run OCR on temp file
            result = self.ocr_engine.ocr(str(temp_img_path), cls=True)
            
            # Extract text from result
            if result and result[0]:
                lines = []
                for line in result[0]:
                    if line and len(line) >= 2:
                        text, confidence = line[1]
                        lines.append(text)
                return '\n'.join(lines)
            else:
                return ""
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_img_path)
            except:  # noqa: E722
                pass
    
    @property
    def is_available(self) -> bool:
        """Check if PaddleOCR is available"""
        return PADDLEOCR_AVAILABLE

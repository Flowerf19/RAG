import re
from typing import Optional


class TextCleaner:
    """
    Utility class for cleaning text before embedding to improve quality.
    
    Handles:
    - Extra whitespace normalization
    - Removal of watermark patterns (e.g., "XXX-XXX")
    - Optional lowercase conversion
    """
    
    def __init__(self, remove_watermarks: bool = True, normalize_case: bool = False):
        self.remove_watermarks = remove_watermarks
        self.normalize_case = normalize_case
        
        # Common watermark patterns
        self.watermark_patterns = [
            r'XXX-XXX',  # Remove all XXX-XXX occurrences
        ]
    
    def clean(self, text: str) -> str:
        """
        Clean the input text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove watermark patterns
        if self.remove_watermarks:
            for pattern in self.watermark_patterns:
                text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL)
                # Remove multiple times to handle adjacent occurrences
                while re.search(pattern, text, flags=re.MULTILINE | re.DOTALL):
                    text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL)
        
        # Normalize whitespace: collapse all whitespace to single spaces and remove excessive newlines
        text = ' '.join(text.split())  # This removes all extra spaces, tabs, newlines
        
        # Optional: convert to lowercase
        if self.normalize_case:
            text = text.lower()
        
        return text
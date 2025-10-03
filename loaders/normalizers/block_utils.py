"""
Utilities để lọc và xử lý block trích xuất từ PDF.
"""
from typing import List, Optional, Tuple, Set
import re
import hashlib

def compute_block_hash(text: str) -> str:
    """
    Tính hash của block text để phát hiện block lặp lại.
    """
    if not text:
        return ""
    # Normalize trước khi hash để tăng khả năng match
    normalized = text.strip().lower()
    normalized = re.sub(r'\s+', ' ', normalized)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def is_repeated_block(text: str, block_hash_counter: dict, threshold: int = 3) -> bool:
    """
    Kiểm tra block có lặp lại nhiều lần trong document hay không.
    
    Args:
        text: Nội dung block
        block_hash_counter: Dict đếm số lần xuất hiện của mỗi hash
        threshold: Ngưỡng số lần lặp lại để coi là header/footer (mặc định 3)
    
    Returns:
        True nếu block lặp lại >= threshold lần
    """
    if not text or len(text.strip()) < 5:
        return False
    
    block_hash = compute_block_hash(text)
    if not block_hash:
        return False
    
    # Count occurrences
    count = block_hash_counter.get(block_hash, 0)
    
    # Consider as repeated if appears >= threshold times
    return count >= threshold

def is_header_footer_block(text: str, bbox: Optional[Tuple] = None, page_height: float = 792.0) -> bool:
    """
    Kiểm tra xem block có phải là header/footer dựa vào vị trí và độ ngắn.
    
    Args:
        text: Nội dung block
        bbox: Bounding box (x0, y0, x1, y1)
        page_height: Chiều cao trang (mặc định 792 cho Letter size)
    
    Returns:
        True nếu là header/footer
    """
    if not text or len(text.strip()) < 5:
        return False
    
    # Check if text is very short (likely metadata)
    text_stripped = text.strip()
    if len(text_stripped) < 20 and len(text_stripped.split()) <= 5:
        # Check if in header/footer position
        if bbox and isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
            y0, y1 = bbox[1], bbox[3]
            # Top 10% or bottom 10% of page
            if y0 < page_height * 0.1 or y1 > page_height * 0.9:
                return True
    
    return False

def is_page_number_block(text: str) -> bool:
    """
    Kiểm tra xem block có phải là số trang hay không.
    Pattern: "Page X/Y", "Page X of Y", "X/Y", etc.
    """
    if not text or len(text.strip()) > 30:
        return False
    
    text_stripped = text.strip()
    
    # Pattern matching for page numbers
    page_patterns = [
        r'^\s*page\s+\d+\s*/\s*\d+\s*$',
        r'^\s*page\s+\d+\s+of\s+\d+\s*$',
        r'^\s*\d+\s*/\s*\d+\s*$',
        r'^\s*\[\s*\d+\s*\]\s*$',
    ]
    
    for pattern in page_patterns:
        if re.match(pattern, text_stripped, re.IGNORECASE):
            return True
    
    return False

def is_empty_or_whitespace_block(text: str, min_length: int = 3) -> bool:
    """
    Kiểm tra xem block có rỗng hoặc chỉ chứa whitespace hay không.
    """
    if not text:
        return True
    
    text_stripped = text.strip()
    
    # Check if only whitespace/newline
    if not text_stripped:
        return True
    
    # Check if too short after cleaning
    if len(text_stripped) < min_length:
        return True
    
    return False

def is_bbox_too_small(bbox: Optional[Tuple], min_area: float = 10.0, min_width: float = 5.0, min_height: float = 3.0) -> bool:
    """
    Kiểm tra xem bbox có quá nhỏ (noise) hay không.
    """
    if not bbox or not isinstance(bbox, (tuple, list)) or len(bbox) < 4:
        return False
    
    try:
        x0, y0, x1, y1 = bbox[:4]
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        area = width * height
        
        if area < min_area or width < min_width or height < min_height:
            return True
    except (ValueError, TypeError):
        return False
    
    return False

def normalize_whitespace(text: str) -> str:
    """
    Chuẩn hóa whitespace: multiple spaces -> single space, multiple newlines -> max 2 newlines.
    """
    if not text:
        return text
    
    # Normalize multiple spaces to single space (không áp dụng cho newline)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize multiple newlines to max 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def should_filter_block(
    text: str, 
    bbox: Optional[Tuple], 
    config: Optional[dict] = None,
    block_hash_counter: Optional[dict] = None,
    page_height: float = 792.0
) -> bool:
    """
    Tổng hợp tất cả các điều kiện lọc block.
    Trả về True nếu block nên bị loại bỏ.
    
    Args:
        text: Nội dung block
        bbox: Bounding box
        config: Cấu hình filter
        block_hash_counter: Dict đếm số lần xuất hiện của block (để phát hiện lặp)
        page_height: Chiều cao trang
    """
    if config is None:
        config = {}
    
    # Get config values
    enable_repeated_filter = config.get('enable_repeated_block_filter', True)
    enable_position_filter = config.get('enable_position_filter', True)
    enable_page_number_filter = config.get('enable_page_number_filter', True)
    enable_empty_filter = config.get('enable_empty_filter', True)
    enable_bbox_filter = config.get('enable_bbox_filter', True)
    
    min_text_length = config.get('min_text_length', 3)
    min_bbox_area = config.get('min_bbox_area', 10.0)
    repeated_threshold = config.get('repeated_block_threshold', 3)
    
    # Apply filters
    # 1. Check for repeated blocks (header/footer pattern)
    if enable_repeated_filter and block_hash_counter:
        if is_repeated_block(text, block_hash_counter, threshold=repeated_threshold):
            return True
    
    # 2. Check for position-based header/footer
    if enable_position_filter:
        if is_header_footer_block(text, bbox, page_height):
            return True
    
    # 3. Check for page numbers
    if enable_page_number_filter and is_page_number_block(text):
        return True
    
    # 4. Check for empty/whitespace
    if enable_empty_filter and is_empty_or_whitespace_block(text, min_length=min_text_length):
        return True
    
    # 5. Check for bbox too small
    if enable_bbox_filter and is_bbox_too_small(bbox, min_area=min_bbox_area):
        return True
    
    return False

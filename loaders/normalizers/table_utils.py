"""
Utilities để lọc và xử lý bảng trích xuất từ PDF.
"""
from typing import List, Optional
import hashlib

def is_header_footer_table(table: List[List[str]], threshold: int = 3) -> bool:
    """
    Kiểm tra xem bảng có phải là header/footer lặp lại hay không.
    Dựa vào:
    - Số hàng nhỏ (<=3)
    - Nhiều ô trống
    - Chứa các keyword như "Classification", "Owner", "Company", "Version"
    """
    if not table or len(table) > threshold:
        return False
    
    # Flatten all cells
    all_text = ' '.join(' '.join(row) for row in table).lower()
    
    # Check for common header/footer keywords
    header_keywords = ['classification', 'owner', 'company', 'version', 'isms', 'qms']
    keyword_count = sum(1 for kw in header_keywords if kw in all_text)
    
    # If contains multiple keywords and is short, likely a header/footer
    if keyword_count >= 2 and len(table) <= 3:
        return True
    
    return False

def remove_empty_columns(table: List[List[str]]) -> List[List[str]]:
    """
    Loại bỏ các cột hoàn toàn trống khỏi bảng.
    """
    if not table or not table[0]:
        return table
    
    max_cols = max(len(row) for row in table)
    
    # Identify non-empty columns
    non_empty_cols = []
    for col_idx in range(max_cols):
        has_content = False
        for row in table:
            if col_idx < len(row) and row[col_idx].strip():
                has_content = True
                break
        if has_content:
            non_empty_cols.append(col_idx)
    
    if not non_empty_cols:
        return table
    
    # Rebuild table with only non-empty columns
    result = []
    for row in table:
        new_row = [row[col_idx] if col_idx < len(row) else '' for col_idx in non_empty_cols]
        result.append(new_row)
    
    return result

def remove_empty_rows(table: List[List[str]]) -> List[List[str]]:
    """
    Loại bỏ các hàng hoàn toàn trống khỏi bảng.
    """
    if not table:
        return table
    
    result = []
    for row in table:
        if any(cell.strip() for cell in row):
            result.append(row)
    
    return result

def compute_table_hash(table: List[List[str]]) -> str:
    """
    Tính hash của bảng để phát hiện các bảng trùng lặp.
    """
    content = '|'.join(','.join(row) for row in table)
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def filter_duplicate_tables(tables: List[List[List[str]]]) -> List[List[List[str]]]:
    """
    Loại bỏ các bảng trùng lặp dựa trên hash.
    """
    if not tables:
        return tables
    
    seen_hashes = set()
    result = []
    
    for table in tables:
        table_hash = compute_table_hash(table)
        if table_hash not in seen_hashes:
            seen_hashes.add(table_hash)
            result.append(table)
    
    return result

def clean_table(table: List[List[str]]) -> Optional[List[List[str]]]:
    """
    Làm sạch bảng: loại bỏ hàng/cột trống, kiểm tra header/footer.
    Trả về None nếu bảng nên bị loại bỏ.
    """
    if not table:
        return None
    
    # Check if it's a header/footer
    if is_header_footer_table(table):
        return None
    
    # Remove empty rows and columns
    table = remove_empty_rows(table)
    table = remove_empty_columns(table)
    
    # After cleaning, check if table still has content
    if not table or len(table) < 1:
        return None
    
    # Check if table has at least 2 columns (meaningful table)
    if table and max(len(row) for row in table) < 2:
        return None
    
    return table

def clean_tables(tables: List[List[List[str]]]) -> List[List[List[str]]]:
    """
    Làm sạch danh sách bảng: loại bỏ header/footer, hàng/cột trống, bảng trùng lặp.
    """
    if not tables:
        return []
    
    # Clean each table
    cleaned = []
    for table in tables:
        clean = clean_table(table)
        if clean is not None:
            cleaned.append(clean)
    
    # Remove duplicates
    cleaned = filter_duplicate_tables(cleaned)
    
    return cleaned

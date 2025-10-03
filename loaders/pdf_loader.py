import os
import logging
import hashlib
from dataclasses import asdict
from typing import Any, Optional, List, Dict, Tuple
import re
import fitz  # PyMuPDF
import pdfplumber

from .config import load_preprocessing_config
from .model.page import PDFPage
from .model.document import PDFDocument
from .model.table import TableSchema, TableRow, TableCell
from .model.block import Block

logger = logging.getLogger("PDFLoader")

def _extract_leading_number(value: str) -> Optional[int]:
    if not isinstance(value, str):
        return None
    
    match = re.match(r'^\s*(\d+)', value)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None

def _header_looks_like_continuation(tbl: TableSchema) -> bool:
    header = getattr(tbl, 'header', None) or []
    rows = getattr(tbl, 'rows', None) or []
    if not rows:
        return False
    first_num = None
    if header and header[0].strip():
        first_num = _extract_leading_number(header[0])
    if first_num is None:
        first_row_value = ''
        first_row = rows[0]
        if getattr(first_row, 'cells', None):
            first_row_value = first_row.cells[0].value or ''
        first_num = _extract_leading_number(first_row_value)
    if first_num is None:
        return False
    next_num = None
    for row in rows:
        cells = getattr(row, 'cells', None) or []
        if not cells:
            continue
        candidate = _extract_leading_number(cells[0].value or '')
        if candidate is not None:
            next_num = candidate
            break
    if next_num is None:
        return first_num > 1
    if next_num == first_num or next_num == first_num + 1:
        return True
    if first_num > 1 and next_num > first_num:
        return True
    return False

def _make_row(values: List[str], row_idx: int) -> TableRow:
    cells = []
    for col_idx, val in enumerate(values, start=1):
        cells.append(TableCell(value=val, row=row_idx, col=col_idx, bbox=None, metadata={}))
    return TableRow(cells=cells, row_idx=row_idx)

def _reindex_rows(rows: List[TableRow]) -> None:
    for r_idx, row in enumerate(rows, start=1):
        row.row_idx = r_idx
        for c_idx, cell in enumerate(row.cells, start=1):
            cell.row = r_idx
            cell.col = c_idx

def _match_row_to_columns(row: TableRow, target_len: int) -> None:
    cells = list(getattr(row, "cells", []) or [])
    if target_len <= 0:
        return
    if len(cells) > target_len:
        merged = " ".join(cell.value for cell in cells[target_len-1:]).strip()
        new_cells = cells[:target_len-1] + [TableCell(value=merged, row=row.row_idx, col=target_len, bbox=None, metadata={})]
    elif len(cells) < target_len:
        new_cells = cells + [TableCell(value='', row=row.row_idx, col=len(cells)+i+1, bbox=None, metadata={}) for i in range(target_len - len(cells))]
    else:
        new_cells = cells
    for idx, cell in enumerate(new_cells, start=1):
        cell.row = row.row_idx
        cell.col = idx
    row.cells = new_cells

def _rebuild_markdown(header: List[str], rows: List[TableRow]) -> str:
    md_lines: List[str] = []
    if header:
        md_lines.append('| ' + ' | '.join(header) + ' |')
        md_lines.append('|' + ('---|' * len(header)))
    for row in rows:
        md_lines.append('| ' + ' | '.join(cell.value for cell in row.cells) + ' |')
    return '\n'.join(md_lines) + ('\n' if md_lines else '')

class PDFLoader:
    def load(self, file_path: str) -> PDFDocument:
        """
        Wrapper cho load_pdf để tương thích với pipeline.
        """
        return self.load_pdf(file_path)
    """
    PDF Loader class - chỉ chịu trách nhiệm load và parse PDF thành structured data.
    KHÔNG bao gồm normalize hay chunking logic.
    """
    def __init__(self, config_path: Optional[str] = None) -> None:
        try:
            self.config = load_preprocessing_config()
        except Exception as e:
            logger.error("Failed to load preprocessing config: %s", e)
            raise RuntimeError(f"PDFLoader config loading failed: {e}")
        self.extract_text: bool = bool(self.config.get('extract_text', True))
        self.extract_tables: bool = bool(self.config.get('extract_tables', True))
        self.tables_engine: str = str(self.config.get('tables_engine', 'auto')).lower()
        if 'min_repeated_text_threshold' not in self.config:
            logger.error("Missing required config: min_repeated_text_threshold")
            raise ValueError("Missing required config: min_repeated_text_threshold")
        self.min_repeated_text_threshold: int = int(self.config['min_repeated_text_threshold'])
        logger.info("PDFLoader initialized with config: extract_text=%s, extract_tables=%s, engine=%s, min_repeated_text_threshold=%s", 
                   self.extract_text, self.extract_tables, self.tables_engine, self.min_repeated_text_threshold)

    def _file_sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()

    def _extract_tables_for_page(self, file_path: str, page_num_1based: int, plumber_pdf: Optional[Any]) -> List[Dict[str, Any]]:
        camelot_module = None
        try:
            import camelot as _camelot
            camelot_module = _camelot
        except Exception:
            camelot_module = None
        try:
            return TableSchema.extract_tables_for_page(
                file_path=file_path,
                page_num_1based=page_num_1based,
                plumber_pdf=plumber_pdf,
                tables_engine=self.tables_engine,
                camelot_module=camelot_module,
            )
        except Exception:
            return []

    def _open_documents(self, file_path: str) -> Tuple[Optional[Any], Optional[Any], List[str]]:
        file_warnings: List[str] = []
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            file_warnings.append(f'fitz open failed: {e}')
            return None, None, file_warnings
        plumber_pdf: Optional[Any] = None
        if self.extract_tables:
            try:
                plumber_pdf = pdfplumber.open(file_path)
            except Exception as e:
                file_warnings.append(f'pdfplumber open failed: {e}')
                plumber_pdf = None
        return doc, plumber_pdf, file_warnings

    def assign_table_captions(self, table_objs: List[TableSchema], file_path: str) -> None:
        """
        Gán caption cho bảng bằng cách sử dụng pdfplumber để trích xuất text với thông tin chi tiết.
        Tìm text có pattern "Table X.X" trong vùng gần bảng, ưu tiên text có font size lớn.
        Đảm bảo mỗi caption chỉ được dùng 1 lần trên cùng 1 trang.
        """
        from collections import defaultdict
        
        table_regex = re.compile(r"table\s*\d+(\.\d+)*", re.IGNORECASE)
        
        # Track captions đã được sử dụng trên mỗi trang
        used_captions = defaultdict(set)
        
        # Mở PDF với pdfplumber để lấy chi tiết text
        try:
            with pdfplumber.open(file_path) as pdf:
                for t in table_objs:
                    if not hasattr(t, 'bbox') or not t.bbox or not hasattr(t, 'page_number'):
                        continue
                    page_num = t.page_number
                    bbox = t.bbox
                    if not bbox or len(bbox) != 4:
                        continue
                    
                    # pdfplumber page index is 0-based
                    if page_num - 1 >= len(pdf.pages):
                        continue
                    page = pdf.pages[page_num - 1]
                    
                    x0, y0, x1, y1 = bbox
                    
                    # Lấy tất cả text objects trong trang với thông tin chi tiết
                    words = page.extract_words(keep_blank_chars=False, x_tolerance=3, y_tolerance=3)
                    
                    # Ghép các words cùng dòng để tạo thành text hoàn chỉnh
                    lines = defaultdict(list)
                    for word in words:
                        y_key = round(word['top'] / 2) * 2  # Group by 2-pixel increments
                        lines[y_key].append(word)
                    
                    # Tìm các dòng có pattern "Table X.X"
                    table_title_candidates = []
                    for y_key, line_words in lines.items():
                        # Sắp xếp words theo x
                        line_words.sort(key=lambda w: w['x0'])
                        # Ghép text
                        line_text = ' '.join(w.get('text', '') for w in line_words)
                        
                        if table_regex.search(line_text):
                            # Lấy bbox của cả dòng
                            wx0 = min(w['x0'] for w in line_words)
                            wy0 = min(w['top'] for w in line_words)
                            wx1 = max(w['x1'] for w in line_words)
                            wy1 = max(w['bottom'] for w in line_words)
                            
                            # Tính khoảng cách đến bảng (kiểm tra cả phía trên và phía dưới)
                            position = None
                            dist_y = None
                            if wy1 <= y0:  # Phía trên bảng
                                dist_y = y0 - wy1
                                position = 'above'
                            elif wy0 >= y1:  # Phía dưới bảng
                                dist_y = wy0 - y1
                                position = 'below'
                            else:  # Chồng lấn với bảng
                                continue
                            
                            dist_x = 0
                            if wx1 < x0:
                                dist_x = x0 - wx1
                            elif wx0 > x1:
                                dist_x = wx0 - x1
                            
                            font_size = max(w.get('height', 10) for w in line_words)
                            table_title_candidates.append({
                                'text': line_text.strip(),
                                'x0': wx0, 'y0': wy0, 'x1': wx1, 'y1': wy1,
                                'dist_y': dist_y,
                                'dist_x': dist_x,
                                'font_size': font_size,
                                'position': position
                            })
                    
                    if not table_title_candidates:
                        # Fallback: tìm dòng text có font size lớn gần bảng (có thể là tiêu đề không chuẩn)
                        fallback_candidates = []
                        for y_key, line_words in lines.items():
                            line_words.sort(key=lambda w: w['x0'])
                            line_text = ' '.join(w.get('text', '') for w in line_words).strip()
                            
                            if not line_text or len(line_text) < 5:
                                continue
                            
                            wx0 = min(w['x0'] for w in line_words)
                            wy0 = min(w['top'] for w in line_words)
                            wx1 = max(w['x1'] for w in line_words)
                            wy1 = max(w['bottom'] for w in line_words)
                            
                            position = None
                            dist_y = None
                            if wy1 <= y0:
                                dist_y = y0 - wy1
                                position = 'above'
                            elif wy0 >= y1:
                                dist_y = wy0 - y1
                                position = 'below'
                            else:
                                continue
                            
                            dist_x = 0
                            if wx1 < x0:
                                dist_x = x0 - wx1
                            elif wx0 > x1:
                                dist_x = wx0 - x1
                            
                            font_size = max(w.get('height', 10) for w in line_words)
                            if dist_y < 50 and dist_x < 100 and font_size >= 9:
                                fallback_candidates.append({
                                    'text': line_text,
                                    'dist_y': dist_y,
                                    'dist_x': dist_x,
                                    'font_size': font_size,
                                    'position': position
                                })
                        
                        if fallback_candidates:
                            below = [c for c in fallback_candidates if c['position'] == 'below']
                            above = [c for c in fallback_candidates if c['position'] == 'above']
                            
                            if below:
                                below.sort(key=lambda x: (x['dist_y'], -x['font_size']))
                                caption = below[0]['text']
                            elif above:
                                above.sort(key=lambda x: (x['dist_y'], -x['font_size']))
                                caption = above[0]['text']
                            else:
                                continue
                            
                            if caption:
                                if not hasattr(t, 'metadata') or t.metadata is None:
                                    t.metadata = {}
                                t.metadata['table_caption'] = caption
                        continue
                    
                    # Chọn candidate tốt nhất: ưu tiên phía dưới, sau đó gần nhất, rồi font_size lớn
                    # Loại bỏ caption đã được sử dụng trên cùng trang này
                    below_candidates = [c for c in table_title_candidates 
                                      if c['position'] == 'below' and c['text'] not in used_captions[page_num]]
                    above_candidates = [c for c in table_title_candidates 
                                      if c['position'] == 'above' and c['text'] not in used_captions[page_num]]
                    
                    best_candidate = None
                    if below_candidates:
                        below_candidates.sort(key=lambda x: (x['dist_y'], x['dist_x'], -x['font_size']))
                        best_candidate = below_candidates[0]
                    elif above_candidates:
                        above_candidates.sort(key=lambda x: (x['dist_y'], x['dist_x'], -x['font_size']))
                        best_candidate = above_candidates[0]
                    
                    if best_candidate:
                        caption = best_candidate['text']
                        if caption:
                            if not hasattr(t, 'metadata') or t.metadata is None:
                                t.metadata = {}
                            t.metadata['table_caption'] = caption
                            # Mark caption as used on this page
                            used_captions[page_num].add(caption)
        except Exception as e:
            logger.warning(f"Caption assignment failed: {e}")

    def load_pdf(self, file_path: str) -> PDFDocument:
        pages: List[PDFPage] = []
        doc_id = os.path.basename(file_path)
        doc_title, page_labels, meta, num_pages = PDFDocument.extract_metadata(file_path)
        doc, plumber_pdf, file_warnings = self._open_documents(file_path)
        if doc is None:
            return PDFDocument(
                file_path=file_path,
                num_pages=num_pages,
                meta=meta,
                pages=[],
                warnings=file_warnings
            )
        all_blocks = PDFDocument.collect_all_blocks(doc)
        
        # Collect block_hash_counter to detect repeated blocks across document
        from loaders.normalizers.block_utils import compute_block_hash
        from collections import Counter
        block_hash_counter = Counter()
        for blocks_in_page in all_blocks:
            for block_tuple in blocks_in_page:
                # block_tuple: (x0, y0, x1, y1, text, block_no, block_type)
                if len(block_tuple) >= 5:
                    text = block_tuple[4]
                    if text and len(text.strip()) >= 5:
                        block_hash = compute_block_hash(text)
                        if block_hash:
                            block_hash_counter[block_hash] += 1
        
        # Không normalize, chỉ trả về block thô, nhưng luôn trích xuất bảng nếu extract_tables
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            blocks = all_blocks[page_idx] if all_blocks and page_idx < len(all_blocks) else []
            page_meta = {
                "file_path": file_path,
                "page_number": page_idx + 1,
                "doc_id": doc_id,
                "doc_title": doc_title,
                "page_label": page_labels.get(page_idx) if isinstance(page_labels, dict) else None,
                "page_size": {"width": float(page.rect.width), "height": float(page.rect.height)},
            }
            tables = []
            if self.extract_tables:
                try:
                    raw_tables = self._extract_tables_for_page(file_path, page_idx + 1, plumber_pdf)
                    # raw_tables is now List[Dict[str, Any]] with 'matrix' and 'bbox'
                    # Clean tables: remove header/footer, empty rows/cols, duplicates
                    from loaders.normalizers.table_utils import clean_tables
                    # Extract just the matrix for cleaning
                    matrices = [t.get('matrix', t) if isinstance(t, dict) else t for t in raw_tables]
                    cleaned_matrices = clean_tables(matrices)
                    # Rebuild tables with bbox if available
                    tables = []
                    for idx, clean_mat in enumerate(cleaned_matrices):
                        bbox = None
                        if idx < len(raw_tables) and isinstance(raw_tables[idx], dict):
                            bbox = raw_tables[idx].get('bbox')
                        tables.append({'matrix': clean_mat, 'bbox': bbox})
                except Exception as e:
                    logger.warning(f"Table extraction failed for page {page_idx+1}: {e}")
            pages.append(PDFPage(
                page_number=page_idx + 1,
                text="",  # text sẽ được tạo trong xử lý nếu cần
                blocks=blocks,
                tables=tables,
                warnings=[],
                source=page_meta,
            ))
        # Sau khi load tất cả pages, convert tables thành TableSchema, merge, và gán caption
        if self.extract_tables:
            try:
                # Thu thập tất cả TableSchema từ các trang
                all_table_objs = []
                for page in pages:
                    for t in page.tables:
                        if isinstance(t, TableSchema):
                            all_table_objs.append(t)
                        elif isinstance(t, dict) and 'matrix' in t:
                            matrix = t.get('matrix', [])
                            bbox = t.get('bbox')
                            if matrix:
                                t_obj = TableSchema.from_matrix(matrix, file_path=file_path, page_number=page.page_number, bbox=bbox)
                                all_table_objs.append(t_obj)
                
                # Merge tables bị split trước
                merged_tables = TableSchema.merge_split_tables(all_table_objs)
                
                # Gán caption sau khi merge
                self.assign_table_captions(merged_tables, file_path)
                
                # Cập nhật lại tables vào pages (keep dict format với matrix + bbox)
                # Group merged tables by page
                from collections import defaultdict
                tables_by_page = defaultdict(list)
                for t in merged_tables:
                    tables_by_page[t.page_number].append(t)
                
                # Update pages với merged + captioned tables
                for page in pages:
                    if page.page_number in tables_by_page:
                        page.tables = [{'matrix': t.to_matrix(), 'bbox': t.bbox} if hasattr(t, 'to_matrix') 
                                      else {'matrix': [[]], 'bbox': t.bbox} for t in tables_by_page[page.page_number]]
                        # Also attach metadata with caption
                        for i, t in enumerate(tables_by_page[page.page_number]):
                            if i < len(page.tables) and hasattr(t, 'metadata') and t.metadata:
                                if isinstance(page.tables[i], dict):
                                    page.tables[i]['metadata'] = t.metadata
            except Exception as e:
                logger.warning(f"Table merge/caption assignment failed: {e}")
        
        try:
            if plumber_pdf is not None:
                plumber_pdf.close()
        except Exception:
            pass
        return PDFDocument(
            file_path=file_path,
            num_pages=num_pages if num_pages else doc.page_count,
            meta=meta,
            pages=pages,
            warnings=file_warnings,
            repeated_block_hashes=set()
        )

    def load_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        pdf_files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith('.pdf')
        ]
        return [asdict(self.load_pdf(f)) for f in pdf_files]

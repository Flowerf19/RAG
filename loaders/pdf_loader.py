import os
import logging
import hashlib
from dataclasses import asdict
from typing import Any, Optional, List, Dict, Tuple
import re
import fitz  # PyMuPDF
import pdfplumber

from .model.page import PDFPage
from .model.document import PDFDocument
from .model.table import TableSchema, TableRow, TableCell
from .model.block import Block
from .normalizers.block_utils import should_filter_block, merge_blocks_list, analyze_block_improvement

logger = logging.getLogger("PDFLoader")


class PDFLoader:
    """
    PDF Loader class - chỉ chịu trách nhiệm load và parse PDF thành structured data.
    KHÔNG bao gồm normalize hay chunking logic.
    
    Tất cả các thuộc tính cấu hình được triển khai trực tiếp trong class theo chuẩn OOP.
    """
    
    def __init__(
        self,
        extract_text: bool = True,
        extract_tables: bool = True,
        tables_engine: str = "auto",
        min_repeated_text_threshold: int = 3,
        min_text_length: int = 10,
        repeated_block_threshold: int = 3,
        enable_repeated_block_filter: bool = True,
        enable_position_filter: bool = True,
        enable_page_number_filter: bool = True,
        enable_empty_filter: bool = True,
        enable_bbox_filter: bool = True,
        min_bbox_area: float = 10.0,
        enable_block_merging: bool = True,
        min_block_length: int = 50
    ) -> None:
        """
        Khởi tạo PDFLoader với các thuộc tính cấu hình.
        
        Args:
            extract_text: Có trích xuất text không
            extract_tables: Có trích xuất bảng không  
            tables_engine: Engine để trích xuất bảng ('auto', 'camelot', 'pdfplumber')
            min_repeated_text_threshold: Ngưỡng tối thiểu để phát hiện text lặp lại
            min_text_length: Độ dài text tối thiểu cho block hợp lệ
            repeated_block_threshold: Block lặp lại >= ngưỡng này sẽ bị lọc
            enable_repeated_block_filter: Bật bộ lọc block lặp lại
            enable_position_filter: Bật bộ lọc vị trí
            enable_page_number_filter: Bật bộ lọc số trang
            enable_empty_filter: Bật bộ lọc block rỗng
            enable_bbox_filter: Bật bộ lọc bbox
            min_bbox_area: Diện tích bbox tối thiểu
            enable_block_merging: Bật tính năng merge blocks phân mảnh
            min_block_length: Độ dài tối thiểu để block được coi là "short"
        """
        # Core extraction settings
        self.extract_text: bool = extract_text
        self.extract_tables: bool = extract_tables
        self.tables_engine: str = tables_engine.lower()
        
        # Block filtering settings
        self.min_repeated_text_threshold: int = min_repeated_text_threshold
        self.min_text_length: int = min_text_length
        self.repeated_block_threshold: int = repeated_block_threshold
        
        # Filter enablement flags
        self.enable_repeated_block_filter: bool = enable_repeated_block_filter
        self.enable_position_filter: bool = enable_position_filter
        self.enable_page_number_filter: bool = enable_page_number_filter
        self.enable_empty_filter: bool = enable_empty_filter
        self.enable_bbox_filter: bool = enable_bbox_filter
        
        # Bbox settings
        self.min_bbox_area: float = min_bbox_area
        
        # Block merging settings
        self.enable_block_merging: bool = enable_block_merging
        self.min_block_length: int = min_block_length
        
        # Validate settings
        self._validate_config()
        
        logger.info(
            "PDFLoader initialized: extract_text=%s, extract_tables=%s, engine=%s, "
            "min_repeated_text_threshold=%s, min_text_length=%s", 
            self.extract_text, self.extract_tables, self.tables_engine,
            self.min_repeated_text_threshold, self.min_text_length
        )
    
    def _validate_config(self) -> None:
        """Validate cấu hình đầu vào."""
        if self.min_repeated_text_threshold < 1:
            raise ValueError("min_repeated_text_threshold must be >= 1")
        if self.min_text_length < 0:
            raise ValueError("min_text_length must be >= 0")
        if self.repeated_block_threshold < 1:
            raise ValueError("repeated_block_threshold must be >= 1")
        if self.min_bbox_area < 0:
            raise ValueError("min_bbox_area must be >= 0")
        if self.tables_engine not in ('auto', 'camelot', 'pdfplumber'):
            logger.warning("Unknown tables_engine '%s', falling back to 'auto'", self.tables_engine)
            self.tables_engine = 'auto'

    @classmethod
    def create_default(cls) -> 'PDFLoader':
        """
        Factory method để tạo PDFLoader với cấu hình mặc định.
        Equivalent với config cũ từ YAML.
        """
        return cls(
            extract_text=True,
            extract_tables=True,
            tables_engine="auto",
            min_repeated_text_threshold=3,
            min_text_length=10,
            repeated_block_threshold=3,
            enable_repeated_block_filter=True,
            enable_position_filter=True,
            enable_page_number_filter=True,
            enable_empty_filter=True,
            enable_bbox_filter=True,
            min_bbox_area=10.0,
            enable_block_merging=True,
            min_block_length=50
        )
    
    @classmethod
    def create_text_only(cls) -> 'PDFLoader':
        """
        Factory method để tạo PDFLoader chỉ trích xuất text, không có bảng.
        """
        return cls(
            extract_text=True,
            extract_tables=False,
            tables_engine="auto",
            min_repeated_text_threshold=3,
            min_text_length=10,
            enable_block_merging=True,
            min_block_length=50
        )
    
    @classmethod 
    def create_tables_only(cls) -> 'PDFLoader':
        """
        Factory method để tạo PDFLoader chỉ trích xuất bảng, không có text.
        """
        return cls(
            extract_text=False,
            extract_tables=True,
            tables_engine="auto",
            min_repeated_text_threshold=3,
            enable_block_merging=False,  # Tables only không cần merge blocks
            min_block_length=50
        )

    def load(self, file_path: str) -> PDFDocument:
        """
        Wrapper cho load_pdf để tương thích với pipeline.
        """
        return self.load_pdf(file_path)

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
            
            # Apply block merging if enabled (BEFORE creating page)
            if self.enable_block_merging and blocks:
                merge_config = {
                    'min_block_length': self.min_block_length,
                    'sentence_endings': ('.', '!', '?', ':', ';'),
                    'list_markers': ('•', '-', '○', '*')
                }
                original_count = len(blocks)
                blocks = merge_blocks_list(blocks, merge_config)
                if len(blocks) != original_count:
                    logger.debug(f"Page {page_idx+1}: Merged {original_count} blocks into {len(blocks)} blocks")
            
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
        """Load tất cả PDF files trong một directory."""
        pdf_files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith('.pdf')
        ]
        return [asdict(self.load_pdf(f)) for f in pdf_files]
    
    def get_config(self) -> Dict[str, Any]:
        """
        Xuất cấu hình hiện tại của loader.
        Useful for debugging và logging.
        """
        return {
            'extract_text': self.extract_text,
            'extract_tables': self.extract_tables,
            'tables_engine': self.tables_engine,
            'min_repeated_text_threshold': self.min_repeated_text_threshold,
            'min_text_length': self.min_text_length,
            'repeated_block_threshold': self.repeated_block_threshold,
            'enable_repeated_block_filter': self.enable_repeated_block_filter,
            'enable_position_filter': self.enable_position_filter,
            'enable_page_number_filter': self.enable_page_number_filter,
            'enable_empty_filter': self.enable_empty_filter,
            'enable_bbox_filter': self.enable_bbox_filter,
            'min_bbox_area': self.min_bbox_area
        }
    
    def update_config(self, **kwargs) -> None:
        """
        Cập nhật cấu hình loader runtime.
        
        Args:
            **kwargs: Các thuộc tính cần cập nhật và giá trị mới
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info("Updated %s to %s", key, value)
            else:
                logger.warning("Unknown config parameter: %s", key)
        
        # Re-validate after update
        self._validate_config()
    
    def enable_all_filters(self) -> None:
        """Bật tất cả các bộ lọc."""
        self.enable_repeated_block_filter = True
        self.enable_position_filter = True
        self.enable_page_number_filter = True
        self.enable_empty_filter = True
        self.enable_bbox_filter = True
        logger.info("All filters enabled")
    
    def disable_all_filters(self) -> None:
        """Tắt tất cả các bộ lọc."""
        self.enable_repeated_block_filter = False
        self.enable_position_filter = False
        self.enable_page_number_filter = False
        self.enable_empty_filter = False
        self.enable_bbox_filter = False
        logger.info("All filters disabled")
    
    def __repr__(self) -> str:
        """String representation of the loader configuration."""
        return (
            f"PDFLoader(extract_text={self.extract_text}, "
            f"extract_tables={self.extract_tables}, "
            f"tables_engine='{self.tables_engine}', "
            f"min_repeated_text_threshold={self.min_repeated_text_threshold})"
        )
    
    # ========== STATIC UTILITY METHODS ==========
    
    @staticmethod
    def _extract_leading_number(value: Any) -> Optional[int]:
        """Extract leading number from a string value."""
        if not isinstance(value, str):
            return None
        
        match = re.match(r'^\s*(\d+)', value)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _header_looks_like_continuation(tbl: TableSchema) -> bool:
        """Check if table header looks like a continuation of previous table."""
        header = getattr(tbl, 'header', None) or []
        rows = getattr(tbl, 'rows', None) or []
        if not rows:
            return False
        first_num = None
        if header and header[0].strip():
            first_num = PDFLoader._extract_leading_number(header[0])
        if first_num is None:
            first_row_value = ''
            first_row = rows[0]
            if getattr(first_row, 'cells', None):
                first_row_value = first_row.cells[0].value or ''
            first_num = PDFLoader._extract_leading_number(first_row_value)
        if first_num is None:
            return False
        next_num = None
        for row in rows:
            cells = getattr(row, 'cells', None) or []
            if not cells:
                continue
            candidate = PDFLoader._extract_leading_number(cells[0].value or '')
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

    @staticmethod
    def _make_row(values: List[str], row_idx: int) -> TableRow:
        """Create a TableRow from list of values."""
        cells = []
        for col_idx, val in enumerate(values, start=1):
            cells.append(TableCell(value=val, row=row_idx, col=col_idx, bbox=None, metadata={}))
        return TableRow(cells=cells, row_idx=row_idx)

    @staticmethod
    def _reindex_rows(rows: List[TableRow]) -> None:
        """Reindex row and cell indices for consistency."""
        for r_idx, row in enumerate(rows, start=1):
            row.row_idx = r_idx
            for c_idx, cell in enumerate(row.cells, start=1):
                cell.row = r_idx
                cell.col = c_idx

    @staticmethod
    def _match_row_to_columns(row: TableRow, target_len: int) -> None:
        """Adjust row to have exactly target_len columns."""
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

    @staticmethod
    def _rebuild_markdown(header: List[str], rows: List[TableRow]) -> str:
        """Rebuild markdown table from header and rows."""
        md_lines: List[str] = []
        if header:
            md_lines.append('| ' + ' | '.join(header) + ' |')
            md_lines.append('|' + ('---|' * len(header)))
        for row in rows:
            md_lines.append('| ' + ' | '.join(cell.value for cell in row.cells) + ' |')
        return '\n'.join(md_lines) + ('\n' if md_lines else '')

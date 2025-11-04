# Module `PDFLoaders` — Smart PDF Processing với OCR Integration

Mục tiêu: thư mục `PDFLoaders/` cung cấp hệ thống xử lý PDF thông minh với khả năng tự động phát hiện loại PDF (text-based vs image-based) và tích hợp OCR để trích xuất nội dung từ bảng biểu và hình ảnh. Thiết kế theo nguyên tắc Single Responsibility: mỗi extractor chuyên trách nhiệm riêng (text, tables, figures, OCR).

README này mô tả kiến trúc, API công khai, các extractor có sẵn, ví dụ sử dụng, kiểm thử và các lưu ý vận hành (PaddleOCR, pdfplumber).

## Nội dung thư mục (tóm tắt)

- `pdf_provider.py` — Re-export các class chính từ provider package
- `provider/` — Core PDF processing logic:
  - `pdf_provider.py` — `PDFProvider` class chính với auto-detection logic
  - `simple_provider.py` — `SimpleTextProvider` cho text extraction đơn giản
  - `models.py` — Data models (`PDFDocument`, `PageContent`)
  - `extractors/` — Specialized extractors:
    - `ocr_extractor.py` — `OCRExtractor` với PaddleOCR integration
    - `table_extractor.py` — `TableExtractor` với pdfplumber + OCR enhancement
    - `figure_extractor.py` — `FigureExtractor` với image grouping + OCR
- `configs/` — Cấu hình cho các extractors
- `models/` — Pre-trained models cho OCR
- `pdf_extract_kit/` — Utility tools cho PDF processing
- `requirements/` — Dependencies riêng cho PDF processing

## Contract (inputs / outputs / error modes)

- **Input**: PDF file path (str) hoặc Path object
- **Output**: `PDFDocument` object chứa list `PageContent` (text, tables, figures)
- **Error modes**: File not found, corrupted PDF, OCR failures (graceful degradation)

## Thiết kế & hành vi từng thành phần

### `PDFProvider` (`provider/pdf_provider.py`)

Lớp chính điều phối việc xử lý PDF với logic thông minh:

**Auto-Detection Logic:**
- **Text-based PDF** (>50 chars/page): Sử dụng PyMuPDF text extraction
- **Image-based PDF** (<50 chars/page): Sử dụng PaddleOCR cho toàn bộ trang
- **Mixed PDF**: Hybrid approach per page

**Tính năng:**
- `PDFProvider(use_ocr="auto", ocr_lang="multilingual", min_text_threshold=50)`
- `load(pdf_path) → PDFDocument`
- Tích hợp `OCRExtractor`, `TableExtractor`, `FigureExtractor`

### `OCRExtractor` (`provider/extractors/ocr_extractor.py`)

Xử lý OCR với PaddleOCR engine:

**Language Mapping:**
```python
LANG_MAP = {
    "multilingual": "en",      # Default fallback
    "en": "en", "ch": "ch", "vi": "vi",
    "fr": "latin", "de": "latin", "es": "latin", "it": "latin",
    "pt": "latin", "nl": "latin", "pl": "latin",
    "ru": "cyrillic", "uk": "cyrillic",
    "ar": "arabic", "hi": "devanagari",
    "ja": "japan", "ko": "korean"
}
```

**Tính năng:**
- `OCRExtractor(lang="multilingual")`
- `extract_text(image) → str`
- `extract_page_ocr(page_fitz) → str`
- Hỗ trợ 20+ ngôn ngữ với auto-mapping

### `TableExtractor` (`provider/extractors/table_extractor.py`)

Trích xuất bảng biểu với OCR enhancement:

**Logic Enhancement:**
- Sử dụng pdfplumber để extract tables
- Khi >30% cells trống → trigger OCR enhancement
- Thêm row `[OCR Supplement]` với page OCR text

**Tính năng:**
- `TableExtractor(ocr_extractor=None)`
- `extract_tables(page_fitz, page_num) → List[Dict]`
- `_enhance_with_ocr(table_data, page_ocr_text)` — internal method

### `FigureExtractor` (`provider/extractors/figure_extractor.py`)

Trích xuất hình ảnh/sơ đồ với OCR text:

**Grouping Logic:**
- Group images theo vị trí và kích thước
- Extract OCR text cho từng figure group
- Store trong `figure['text']` field

**Tính năng:**
- `FigureExtractor(ocr_extractor=None)`
- `extract_figures(page_fitz, page_num) → List[Dict]`
- Logging với filename context cho debugging

## Data Models

### `PDFDocument`
```python
@dataclass
class PDFDocument:
    filename: str
    pages: List[PageContent]
    metadata: Dict[str, Any]
```

### `PageContent`
```python
@dataclass
class PageContent:
    page_number: int
    text: str           # Main text content
    tables: List[Dict]  # Extracted tables
    figures: List[Dict] # Extracted figures with OCR text
```

## Ví dụ sử dụng

### Basic Usage
```python
from PDFLoaders import PDFProvider

# Initialize với auto OCR detection
provider = PDFProvider(
    use_ocr="auto",           # "auto", "always", "never"
    ocr_lang="multilingual",  # Language cho OCR
    min_text_threshold=50     # Ngưỡng phát hiện text-based
)

# Load PDF
doc = provider.load("path/to/document.pdf")

# Access content
for page in doc.pages:
    print(f"Page {page.page_number}:")
    print(f"Text: {page.text[:200]}...")
    print(f"Tables: {len(page.tables)}")
    print(f"Figures: {len(page.figures)}")
```

### Advanced Configuration
```python
# Force OCR cho tất cả pages
provider = PDFProvider(use_ocr="always", ocr_lang="vi")

# Disable OCR hoàn toàn
provider = PDFProvider(use_ocr="never")

# Custom threshold
provider = PDFProvider(min_text_threshold=100)
```

### Direct Extractor Usage
```python
from PDFLoaders.provider.extractors import OCRExtractor, TableExtractor

# OCR cho image
ocr = OCRExtractor(lang="en")
text = ocr.extract_text(pil_image)

# Table extraction với OCR enhancement
tables = TableExtractor(ocr_extractor=ocr)
table_data = tables.extract_tables(page_fitz, page_num=1)
```

## Kiểm thử

### Unit Tests
```bash
# Test PDF loading
python -m pytest tests/test_pdf_provider.py -v

# Test OCR extraction
python -m pytest tests/test_ocr_extractor.py -v

# Test table extraction
python -m pytest tests/test_table_extractor.py -v
```

### Integration Tests
```python
# Test với sample PDFs
from PDFLoaders import PDFProvider

provider = PDFProvider()
doc = provider.load("data/pdf/sample.pdf")

# Verify content extraction
assert len(doc.pages) > 0
assert any(page.text for page in doc.pages)
```

## Dependencies & Installation

### Core Dependencies
```txt
PyMuPDF>=1.23.0          # PDF text extraction
pdfplumber>=0.10.0       # Table extraction
PaddleOCR>=2.7.0         # OCR engine
PaddlePaddle>=2.5.0      # PaddleOCR dependency
opencv-python>=4.8.0     # Image processing
Pillow>=10.0.0          # PIL for image handling
```

### Installation
```bash
pip install -r PDFLoaders/requirements/requirements.txt

# Download PaddleOCR models (auto-downloaded on first use)
# Models stored in ~/.paddleocr/
```

## Lưu ý vận hành

### Performance Considerations
- **Memory Usage**: Large PDFs với nhiều images → high memory consumption
- **OCR Speed**: PaddleOCR ~2-5 seconds per page depending on language
- **Caching**: Consider caching OCR results cho repeated processing

### Error Handling
- **OCR Failures**: Graceful fallback to empty string, logged as warnings
- **Corrupted PDFs**: Raises `PDFLoadError` with descriptive message
- **Missing Dependencies**: Clear error messages với installation instructions

### Language Support
- **Primary**: English, Vietnamese, Chinese (CJK)
- **European**: French, German, Spanish, Italian, Portuguese, Dutch, Polish
- **Cyrillic**: Russian, Ukrainian
- **Other**: Arabic, Devanagari (Hindi), Japanese, Korean

### Best Practices
1. **Use Auto-Detection**: `use_ocr="auto"` cho most cases
2. **Language Selection**: Match PDF language với `ocr_lang` parameter
3. **Threshold Tuning**: Adjust `min_text_threshold` cho specific document types
4. **Error Monitoring**: Check logs cho OCR failures và enhancement triggers

## Troubleshooting

### Common Issues

**OCR Not Working:**
```python
# Check PaddleOCR installation
python -c "import paddleocr; print('PaddleOCR OK')"

# Test với simple image
from PDFLoaders.provider.extractors import OCRExtractor
ocr = OCRExtractor(lang="en")
result = ocr.extract_text("path/to/image.png")
```

**Table OCR Enhancement Not Triggering:**
- Check table structure: >30% empty cells required
- Verify OCR extractor properly initialized
- Check logs cho "Enhancing table with OCR"

**Memory Issues với Large PDFs:**
```python
# Process page-by-page thay vì load toàn bộ
provider = PDFProvider()
doc = provider.load("large.pdf")
for page in doc.pages:
    # Process từng page
    process_page_content(page)
```

### Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable PDF provider logging
logger = logging.getLogger('PDFLoaders.provider')
logger.setLevel(logging.DEBUG)
```

## API Reference

### PDFProvider
- `load(pdf_path: Union[str, Path]) → PDFDocument`
- `load_with_progress(pdf_path: Union[str, Path], progress_callback: Callable)`

### OCRExtractor
- `extract_text(image: PIL.Image) → str`
- `extract_page_ocr(page_fitz) → str`
- `test_connection() → bool`

### TableExtractor
- `extract_tables(page_fitz, page_num: int) → List[Dict]`

### FigureExtractor
- `extract_figures(page_fitz, page_num: int) → List[Dict]`

## Contributing

### Adding New Extractors
1. Extend base extractor interface
2. Add to `provider/extractors/__init__.py`
3. Update `PDFProvider` để integrate
4. Add comprehensive tests
5. Update documentation

### Language Support Extension
1. Add language code to `LANG_MAP` in `ocr_extractor.py`
2. Map to appropriate PaddleOCR language
3. Test với sample documents
4. Update documentation</content>
<parameter name="filePath">d:\Project\RAG-2\PDFLoaders\README.md
# RAG Loaders

Module chịu trách nhiệm trích xuất dữ liệu thô từ PDF (loader) và chuẩn hóa/biến đổi (normalizer) qua các class chuyên biệt. Các model trung gian (PDFDocument, PDFPage, Block, TableSchema, ...) đều hỗ trợ mở rộng normalization ở tầng class.

## Cấu trúc

```text
rag/loaders/
├── pdf_loader.py         # Loader: chỉ load và parse PDF, KHÔNG normalize
├── config.py            # YAML config loader
├── ids.py               # ID generation utilities
├── model/               # Data models
│   ├── base.py         # Base classes
│   ├── document.py     # PDFDocument model
│   ├── page.py         # PDFPage model
│   ├── block.py        # Text/Table blocks
│   └── table.py        # Table schema
├── normalizers/         # Data normalization
│   ├── text.py         # Text processing
│   ├── tables.py       # Table processing
│   └── layout.py       # Layout analysis
├── __init__.py         # Package init
└── README.md           # File này
```

## Chức năng chính

### PDFLoader (pdf_loader.py)

- **Chỉ load và parse PDF thành dữ liệu thô**
- KHÔNG thực hiện normalize, KHÔNG xử lý tables, KHÔNG chunking
- Sử dụng PyMuPDF để extract blocks thô cho từng trang
- Trả về các model trung gian (PDFDocument, PDFPage, Block, ...)
- Phù hợp cho pipeline custom hoặc các bước xử lý tiếp theo
- Nếu muốn chuẩn hóa, hãy sử dụng các method `.normalize()` ở từng class model hoặc qua normalizer riêng biệt.

### Data Models (model/)

- **PDFDocument**: Container cho toàn bộ document, có thể mở rộng chuẩn hóa qua `.normalize()`
- **PDFPage**: Đại diện cho một trang PDF, hỗ trợ chuẩn hóa layout/text qua `.normalize()`
- **Block**: Text hoặc table blocks với position info, có thể chuẩn hóa text qua `.normalize()`
- **TableSchema**: Cấu trúc bảng với rows/columns, chuẩn hóa header/rows qua `.normalize()`

### Normalizers (normalizers/)

- **TextNormalizer**: Class/hàm chuẩn hóa text content, có thể dùng độc lập hoặc gọi từ model
- **TableNormalizer**: Chuẩn hóa bảng, header, rows
- **LayoutNormalizer**: Chuẩn hóa vị trí, bbox, reading order

## Loại bỏ dữ liệu trùng lặp, nhiễu, header/footer (Block/Table Filtering)

### Các bước đã thực hiện để làm sạch dữ liệu

#### Đối với Block (text)

- **Chuẩn hóa text ở Block.normalize:**
- Sử dụng `clean-text` và `ftfy` để chuẩn hóa unicode, loại bỏ ký tự vô hình, emoji, ký tự đặc biệt.
- Loại bỏ các chuỗi dấu chấm lặp ("......") thường gặp ở TOC.
- Chuẩn hóa whitespace, loại bỏ nhiều khoảng trắng/thừa dòng.
- Giữ lại line-break hợp lý để phân biệt đoạn/câu.
- **Chuyển đổi block tuple thành Block object trước khi normalize:**
- Đảm bảo mọi block đều được chuẩn hóa text trước khi lọc.
- **Lọc block lặp lại (header/footer):**
- Tính hash cho từng block text toàn document, đếm số lần xuất hiện.
- Nếu một block xuất hiện >= `repeated_block_threshold` (configurable, mặc định 3), block đó sẽ bị loại bỏ (trừ khi là nội dung thực sự dài).
- **Lọc block ngắn/noise:**
- Loại bỏ block có độ dài nhỏ hơn `min_text_length` (configurable, mặc định 10).
- Loại bỏ block chỉ chứa whitespace, số trang, hoặc bbox quá nhỏ.
- **Lọc block theo vị trí (header/footer):**
- Nếu block nằm ở top/bottom của trang và ngắn, sẽ bị loại bỏ.
- **Lọc block trùng lặp cross-document:**
- Các block header/table header/TOC lặp lại ở nhiều file sẽ vẫn còn, nhưng đã loại bỏ phần lớn noise trong từng document.
- **Tất cả tham số lọc đều cấu hình qua YAML (`config/preprocessing.yaml`).**

#### Đối với Table

- **Chuẩn hóa bảng ở TableSchema.normalize:**
- Chuẩn hóa text từng cell, header, row bằng `clean-text` và các rule tương tự block.
- Loại bỏ các dòng/cột trống hoàn toàn.
- Loại bỏ các dòng/cột chỉ chứa ký tự noise (dấu chấm, gạch ngang, ký tự đặc biệt).
- Loại bỏ các dòng/cột lặp lại hoàn toàn trong bảng.
- Chuẩn hóa lại header, merge header nếu bị split.
- **Lọc bảng noise:**
- Bảng chỉ có 1 dòng hoặc 1 cột, hoặc toàn bộ cell trùng lặp sẽ bị loại bỏ.
- Bảng không có giá trị thực (sau khi clean) sẽ bị loại bỏ.
- **Tất cả tham số lọc bảng đều cấu hình qua YAML (`config/preprocessing.yaml`).**

### Kết quả thực nghiệm

- Đã loại bỏ ~50% block noise/trùng lặp trên các file PDF mẫu.
- Đã loại bỏ phần lớn bảng noise, bảng trùng header/footer, bảng chỉ có 1 dòng/cột hoặc toàn ký tự đặc biệt.
- Các block và bảng còn lại chủ yếu là nội dung thực, table, hoặc header/table header cross-document.

## Output Schema

Mỗi chunk/model có thể có:

- `stable_id`: Deterministic ID (hash-based)
- `metadata["citation"]`: Human-readable citation (e.g., "doc-title, p.12")
- `bbox_norm`: Normalized bounding box
- `source`: Full source attribution (doc_id, page_number, etc.)
- `content_sha256`: Content hash for stability

## Config

- `rag/config/preprocessing.yaml`: Cấu hình preprocessing cho loader
- `rag/config/chunking.yaml`: Cấu hình chunking/normalization (nếu cần)

## Cách sử dụng

```python
from rag.loaders.pdf_loader import PDFLoader

# Load PDF thô với cấu hình mặc định
loader = PDFLoader.create_default()
doc = loader.load_pdf("path/to/document.pdf")

# Hoặc với cấu hình tùy chỉnh
loader = PDFLoader(
    extract_text=True,
    extract_tables=True,
    min_repeated_text_threshold=5
)
doc = loader.load_pdf("path/to/document.pdf")

# Chuẩn hóa toàn bộ document (nếu muốn)
doc_norm = doc.normalize()  # Yêu cầu các class model đã implement .normalize()

# Hoặc chuẩn hóa từng page/block
for page in doc.pages:
    page_norm = page.normalize()
```

## Tích hợp

- Input: Raw PDF files từ `data/pdf/`
- Output: `PDFDocument` objects cho `DocumentService`
- Loader chỉ trả về dữ liệu thô, không chunking, không normalize
- Nếu muốn chuẩn hóa, hãy gọi `.normalize()` ở tầng model hoặc dùng normalizer riêng


 Duplicate Statistics:
   • Total blocks: 230
   • Unique texts: 224
   • Duplicate blocks: 6
   • Duplicate rate: 2.6%

✅ LOW DUPLICATE RATE (2.6%)
   Recommendation: Deduplication optional

💡 DEDUPLICATION STRATEGY:
   1. Group blocks by content_sha256
   2. Keep first occurrence per hash
   3. Remove 6 duplicate blocks
   4. Result: 224 unique blocks

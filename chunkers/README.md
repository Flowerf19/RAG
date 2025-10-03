# RAG Chunkers

Module chịu trách nhiệm chia nhỏ (chunking) dữ liệu đã được load và normalize từ module `loaders` thành các chunks phù hợp cho RAG pipeline.

## Cấu trúc

```text
rag/chunkers/
├── chunker.py           # Base chunker và các chunker strategies
├── config.py            # YAML config loader cho chunking
├── ids.py               # ID generation utilities cho chunks
├── model/               # Data models cho chunks
│   ├── base.py         # Base classes
│   ├── chunk.py        # Chunk model
│   ├── metadata.py     # Chunk metadata
│   └── document.py     # ChunkDocument container
├── strategies/          # Chunking strategies
│   ├── fixed_size.py   # Fixed size chunking
│   ├── semantic.py     # Semantic chunking
│   ├── sliding.py      # Sliding window chunking
│   └── hierarchical.py # Hierarchical chunking
├── processors/          # Chunk processing utilities
│   ├── text.py         # Text chunk processing
│   ├── table.py        # Table chunk processing
│   └── hybrid.py       # Hybrid content processing
├── __init__.py         # Package init
└── README.md           # File này
```

## Chức năng chính

### BaseChunker (chunker.py)

- **Interface chung cho tất cả chunking strategies**
- Nhận input từ `PDFDocument` (từ loaders)
- Trả về `ChunkDocument` chứa danh sách các `Chunk`
- Hỗ trợ configuration qua YAML
- Có thể mở rộng với custom strategies

### Chunking Strategies

#### 1. FixedSizeChunker

- Chia document thành chunks có kích thước cố định
- Hỗ trợ overlap giữa các chunks
- Phù hợp cho embedding models có giới hạn token

#### 2. SemanticChunker

- Chia document dựa trên ngữ nghĩa
- Sử dụng sentence boundary, paragraph boundary
- Giữ nguyên ngữ cảnh và ý nghĩa của nội dung

#### 3. SlidingWindowChunker

- Chia document với cửa sổ trượt
- Tạo overlap có kiểm soát giữa các chunks
- Phù hợp cho các trường hợp cần context dày đặc

#### 4. HierarchicalChunker

- Chia document theo cấu trúc phân cấp (heading, section, paragraph)
- Giữ nguyên cấu trúc logic của document
- Phù hợp cho tài liệu có cấu trúc rõ ràng

### Data Models (model/)

- **Chunk**: Đại diện cho một chunk với content, metadata, position info
- **ChunkMetadata**: Metadata của chunk (source, page, position, citations)
- **ChunkDocument**: Container cho tất cả chunks của một document

### Processors (processors/)

- **TextProcessor**: Xử lý text chunks, chuẩn hóa, tính toán token count
- **TableProcessor**: Xử lý table chunks, serialize table thành text/markdown
- **HybridProcessor**: Xử lý chunks có cả text và table

## Configuration

Tất cả các tham số chunking được cấu hình qua `config/chunking.yaml`:

```yaml
chunking:
  strategy: "semantic"  # fixed_size, semantic, sliding_window, hierarchical
  
  # Fixed size chunking
  fixed_size:
    chunk_size: 512        # Số tokens/chars per chunk
    overlap: 50            # Số tokens/chars overlap
    unit: "tokens"         # tokens hoặc chars
  
  # Semantic chunking
  semantic:
    max_chunk_size: 800    # Max size per chunk
    min_chunk_size: 100    # Min size per chunk
    split_on: "sentence"   # sentence, paragraph, section
    preserve_structure: true
  
  # Sliding window chunking
  sliding_window:
    window_size: 512
    stride: 256
    unit: "tokens"
  
  # Hierarchical chunking
  hierarchical:
    levels: ["section", "paragraph", "sentence"]
    max_size_per_level: [2000, 800, 200]
    preserve_hierarchy: true
  
  # Table handling
  table:
    strategy: "separate"   # separate, inline, hybrid
    max_rows_per_chunk: 50
    include_caption: true
    serialize_format: "markdown"  # markdown, csv, json
  
  # General settings
  general:
    min_chunk_size: 50
    max_chunk_size: 2000
    include_metadata: true
    generate_citations: true
```

## Input/Output Schema

### Input (từ loaders)

```python
from loaders import PDFDocument, PDFLoader

loader = PDFLoader()
pdf_doc = loader.load("path/to/file.pdf")
pdf_doc = pdf_doc.normalize()  # Normalize trước khi chunk
```

### Output (chunks)

```python
from chunkers import SemanticChunker

chunker = SemanticChunker()
chunk_doc = chunker.chunk(pdf_doc)

# Mỗi chunk có:
for chunk in chunk_doc.chunks:
    print(chunk.stable_id)          # Deterministic ID
    print(chunk.content)            # Text content
    print(chunk.metadata.citation)  # "doc-title, p.12"
    print(chunk.metadata.source)    # Source file
    print(chunk.metadata.page)      # Page number
    print(chunk.token_count)        # Token count
    print(chunk.char_count)         # Character count
```

## Workflow

1. **Load PDF**: Sử dụng `PDFLoader` để load PDF
2. **Normalize**: Gọi `.normalize()` trên `PDFDocument` để làm sạch
3. **Chunk**: Sử dụng chunker strategy phù hợp
4. **Post-process**: Có thể áp dụng processors để xử lý thêm
5. **Export**: Xuất chunks ra format phù hợp (JSON, CSV, ...)

## Extension Points

- Tạo custom chunking strategy bằng cách extend `BaseChunker`
- Tạo custom processor bằng cách extend base processor classes
- Cấu hình tham số qua YAML config
- Override các method `chunk()`, `process()`, `validate()` trong models

## Testing

- Unit tests cho từng chunking strategy
- Integration tests với loaders
- Performance tests với large documents
- Quality tests để đảm bảo chunks có chất lượng tốt

## Dependencies

- `loaders`: Module load và normalize PDF
- `config`: YAML configuration
- `tiktoken` hoặc `transformers`: Token counting
- `spacy`: Sentence/paragraph boundary detection (recommended)
  - English model: `python -m spacy download en_core_web_sm`
  - Vietnamese model: `python -m spacy download vi_core_news_sm`

## Using spaCy for Better Chunking

Module này hỗ trợ spaCy để cải thiện chất lượng chunking:

### Advantages of spaCy

- **Better sentence boundary detection**: Chính xác hơn regex
- **Multi-language support**: English, Vietnamese, và nhiều ngôn ngữ khác
- **Linguistic features**: Noun phrases, entities, dependencies
- **Fast performance**: C-optimized code

### Installation

```bash
# Install spaCy
pip install spacy

# Download language models
python -m spacy download en_core_web_sm  # English
python -m spacy download vi_core_news_sm  # Vietnamese
```

### Usage with spaCy

```python
from chunkers import SemanticChunker

# Semantic chunking with spaCy (English)
chunker = SemanticChunker({
    "max_chunk_size": 800,
    "min_chunk_size": 100,
    "split_on": "sentence",
    "use_spacy": True,
    "lang": "en"
})

# Vietnamese
chunker_vi = SemanticChunker({
    "use_spacy": True,
    "lang": "vi"
})
```

### Fallback without spaCy

Nếu spaCy không được cài đặt hoặc `use_spacy=False`, module sẽ tự động fallback sang regex-based splitting.

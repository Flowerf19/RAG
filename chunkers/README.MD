# Module `chunkers` — Semantic Text Segmentation với Multi-Language Support

Mục tiêu: thư mục `chunkers/` cung cấp hệ thống chia văn bản thành chunks thông minh sử dụng spaCy cho sentence splitting và coherence scoring. Thiết kế theo nguyên tắc Single Responsibility: chunker chỉ chịu trách nhiệm segmentation, không thực hiện embedding. Hỗ trợ đa ngôn ngữ với auto-selection spaCy models.

README này mô tả kiến trúc, API công khai, các chunker implementations, ví dụ sử dụng, kiểm thử và các lưu ý vận hành (spaCy models, language detection).

## Nội dung thư mục (tóm tắt)

- `semantic_chunker.py` — `SemanticChunker` chính với spaCy integration
- `base_chunker.py` — `BaseChunker` abstract base class
- `config_loader.py` — Configuration loading utilities
- `model/` — Data models cho chunks:
  - `chunk.py` — `Chunk` dataclass
  - `chunk_set.py` — `ChunkSet` collection
  - `chunk_stats.py` — Statistics và metrics
  - `enums.py` — Enums cho chunk types và strategies
  - `provenance_agg.py` — Provenance aggregation
  - `score.py` — Coherence scoring
  - `block_span.py` — Text block spans
- `providers/` — Chunker implementations:
  - `semantic_provider.py` — Semantic chunking logic

## Contract (inputs / outputs / error modes)

- **Input**: `PDFDocument` object từ PDFLoaders
- **Output**: `ChunkSet` chứa list `Chunk` objects với text, provenance, và scores
- **Error modes**: spaCy model not installed, empty documents, encoding issues

## Thiết kế & hành vi từng thành phần

### `SemanticChunker` (`semantic_chunker.py`)

Chunker chính sử dụng spaCy cho sentence segmentation và coherence analysis:

**Language Support (13 ngôn ngữ):**
```python
SPACY_MODEL_MAP = {
    "en": "en_core_web_sm",        # English (small)
    "vi": "vi_core_news_lg",       # Vietnamese (large)
    "zh": "zh_core_web_sm",        # Chinese (small)
    "de": "de_core_news_sm",       # German (small)
    "fr": "fr_core_news_sm",       # French (small)
    "es": "es_core_news_sm",       # Spanish (small)
    "it": "it_core_news_sm",       # Italian (small)
    "pt": "pt_core_news_sm",       # Portuguese (small)
    "nl": "nl_core_news_sm",       # Dutch (small)
    "pl": "pl_core_news_sm",       # Polish (small)
    "ru": "ru_core_news_sm",       # Russian (small)
    "ja": "ja_core_news_sm",       # Japanese (small)
    "multilingual": "xx_ent_wiki_sm",  # Multilingual (small, basic)
}
```

**Coherence Scoring:**
- **Discourse Markers**: Transition words (however, therefore, etc.)
- **Lexical Overlap**: Common words giữa sentences
- **Entity Overlap**: Named entities (optional, requires spaCy NER)

**Tính năng:**
- `SemanticChunker(max_tokens=500, overlap_tokens=50, language="auto")`
- `chunk_document(pdf_doc) → ChunkSet`
- `_aggregate_page_content(page) → str` — combines text + tables + figures
- Auto language detection từ document content

### `BaseChunker` (`base_chunker.py`)

Abstract base class định nghĩa interface cho tất cả chunkers:

**Interface Methods:**
- `chunk_document(pdf_document) → ChunkSet` — main chunking method
- `get_chunk_stats() → ChunkStats` — performance metrics
- `_validate_inputs()` — input validation

### Data Models (`model/`)

#### `Chunk`
```python
@dataclass
class Chunk:
    id: str                    # Unique chunk identifier
    text: str                  # Chunk content
    chunk_type: ChunkType      # SEMANTIC, FIXED_SIZE, etc.
    provenance: ProvenanceAgg  # Source information
    score: Score              # Coherence score
    token_count: int          # Estimated token count
```

#### `ChunkSet`
```python
@dataclass
class ChunkSet:
    chunks: List[Chunk]
    source_document: str
    total_chunks: int
    total_tokens: int
    stats: ChunkStats
```

#### `ProvenanceAgg`
```python
@dataclass
class ProvenanceAgg:
    filename: str
    page_numbers: List[int]
    block_spans: List[BlockSpan]
    content_types: List[str]  # text, table, figure
```

## Ví dụ sử dụng

### Basic Usage
```python
from chunkers import SemanticChunker
from PDFLoaders import PDFProvider

# Load PDF document
pdf_provider = PDFProvider()
pdf_doc = pdf_provider.load("document.pdf")

# Initialize chunker với config (recommended)
chunker = SemanticChunker()  # Auto-load from config/chunker_config.yaml

# Or with custom overrides
chunker = SemanticChunker(
    max_tokens=600,           # Override config max_tokens
    language="vi"             # Force Vietnamese
)

# Chunk document
chunk_set = chunker.chunk_document(pdf_doc)

# Access chunks
for chunk in chunk_set.chunks:
    print(f"Chunk {chunk.id}:")
    print(f"Text: {chunk.text[:200]}...")
    print(f"Pages: {chunk.provenance.page_numbers}")
    print(f"Score: {chunk.score.overall_score}")
```

### Advanced Configuration
```python
# Specify language explicitly
chunker = SemanticChunker(
    max_tokens=300,
    overlap_tokens=30,
    min_sentences_per_chunk=2,
    language="vi"  # Vietnamese
)

# Custom coherence parameters
chunker = SemanticChunker(
    discourse_weight=0.4,     # Weight cho discourse markers
    lexical_weight=0.4,       # Weight cho lexical overlap
    entity_weight=0.2         # Weight cho entity overlap
)
```

### Integration với Pipeline
```python
from pipeline import RAGPipeline

# Pipeline tự động sử dụng SemanticChunker
pipeline = RAGPipeline(
    output_dir="data",
    embedder_type="ollama",
    chunker_config={
        "max_tokens": 400,
        "language": "auto"
    }
)

# Process PDF (auto chunking)
pipeline.process_pdf("document.pdf")
```

## Language Support & Setup

### Supported Languages
- **English** (`en`): `en_core_web_sm`
- **Vietnamese** (`vi`): `vi_core_news_lg` (large model recommended)
- **Chinese** (`zh`): `zh_core_web_sm`
- **European**: German (`de`), French (`fr`), Spanish (`es`), Italian (`it`), Portuguese (`pt`), Dutch (`nl`), Polish (`pl`)
- **Russian** (`ru`): `ru_core_news_sm`
- **Japanese** (`ja`): `ja_core_news_sm`
- **Multilingual** (`multilingual`): `xx_ent_wiki_sm`

### Installing spaCy Models
```bash
# Required models
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
python -c "import spacy; spacy.cli.download('vi_core_news_lg')"

# Optional European languages
python -c "import spacy; spacy.cli.download('de_core_news_sm')"
python -c "import spacy; spacy.cli.download('fr_core_news_sm')"
python -c "import spacy; spacy.cli.download('es_core_news_sm')"

# Check installed models
python -c "import spacy; print(spacy.info())"
```

### Language Auto-Detection
```python
# Auto-detect từ document content
chunker = SemanticChunker(language="auto")

# Chunker sẽ:
# 1. Sample text từ document
# 2. Detect language patterns
# 3. Select appropriate spaCy model
# 4. Fallback to English nếu không detect được
```

## Kiểm thử

### Unit Tests
```bash
# Test chunking logic
python -m pytest tests/test_semantic_chunker.py -v

# Test language detection
python -m pytest tests/test_language_detection.py -v

# Test coherence scoring
python -m pytest tests/test_coherence_scoring.py -v
```

### Integration Tests
```python
# Test với sample document
from chunkers import SemanticChunker
from PDFLoaders import PDFProvider

pdf_provider = PDFProvider()
chunker = SemanticChunker()

# Test Vietnamese document
doc = pdf_provider.load("vietnamese_doc.pdf")
chunks = chunker.chunk_document(doc)

assert len(chunks.chunks) > 0
assert all(chunk.text for chunk in chunks.chunks)
```

### Performance Testing
```python
import time
from chunkers import SemanticChunker

chunker = SemanticChunker(max_tokens=500)
start_time = time.time()

chunks = chunker.chunk_document(large_pdf_doc)
duration = time.time() - start_time

print(f"Chunked {len(chunks.chunks)} chunks in {duration:.2f}s")
print(f"Avg tokens/chunk: {chunks.total_tokens / len(chunks.chunks):.1f}")
```

## Dependencies & Installation

### Core Dependencies
```txt
spacy>=3.7.0            # NLP processing
transformers>=4.21.0    # Tokenization (optional)
numpy>=1.21.0           # Numerical operations
```

### Installation
```bash
pip install -r requirements.txt

# Install spaCy models
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
python -c "import spacy; spacy.cli.download('vi_core_news_lg')"
```

## Lưu ý vận hành

### Performance Considerations
- **Model Loading**: spaCy models ~100-500MB, load once và cache
- **Chunking Speed**: ~100-500 pages/minute depending on model size
- **Memory Usage**: Large documents → consider streaming processing

### Language-Specific Tuning
- **Vietnamese**: Use `vi_core_news_lg` cho tốt nhất accuracy
- **Chinese**: spaCy segmentation works well cho CJK text
- **European**: Small models sufficient, fast processing

### Error Handling
- **Missing Models**: Clear error messages với installation commands
- **Empty Documents**: Returns empty ChunkSet, logged as warning
- **Encoding Issues**: UTF-8 handling với fallback to latin-1

### Best Practices
1. **Language Matching**: Match chunker language với document language
2. **Token Limits**: Adjust `max_tokens` based on embedder limits (512, 1024, etc.)
3. **Overlap Tuning**: 10-20% overlap typically optimal cho coherence
4. **Caching**: Cache chunked results cho repeated processing

## Troubleshooting

### Common Issues

**spaCy Model Not Found:**
```python
# Check installed models
python -c "import spacy; print(spacy.util.get_installed_models())"

# Install missing model
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
```

**Poor Chunk Quality:**
```python
# Use config with language override
chunker = SemanticChunker(language="vi")
chunks = chunker.chunk_document(doc)

for chunk in chunks.chunks:
    print(f"Chunk score: {chunk.score.overall_score}")
    print(f"Discourse: {chunk.score.discourse_score}")
    print(f"Lexical: {chunk.score.lexical_score}")
```

**Language Detection Issues:**
```python
# Force specific language
chunker = SemanticChunker(language="vi")  # Force Vietnamese

# Check language detection
detected = chunker._detect_language(sample_text)
print(f"Detected language: {detected}")
```

**Memory Issues:**
```python
# Process large documents in batches
chunker = SemanticChunker(batch_size=10)  # Process 10 pages at a time

# Or use streaming approach
for page_batch in batch_pages(pdf_doc.pages, batch_size=50):
    chunks = chunker.chunk_pages(page_batch)
    process_chunks(chunks)
```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

chunker = SemanticChunker(debug=True)
chunks = chunker.chunk_document(doc)

# Check chunking statistics
stats = chunks.stats
print(f"Total chunks: {stats.total_chunks}")
print(f"Avg coherence: {stats.avg_coherence_score}")
print(f"Language used: {stats.language_detected}")
```

## API Reference

### SemanticChunker
- `chunk_document(pdf_document) → ChunkSet`
- `chunk_pages(pages) → ChunkSet`
- `_aggregate_page_content(page) → str`
- `_detect_language(text) → str`
- `get_chunk_stats() → ChunkStats`

### BaseChunker (Abstract)
- `chunk_document(pdf_document) → ChunkSet` — *abstract*
- `get_chunk_stats() → ChunkStats` — *abstract*
- `_validate_inputs(pdf_document)` — *abstract*

### Data Models
- `Chunk` — Individual chunk với metadata
- `ChunkSet` — Collection của chunks
- `ChunkStats` — Performance metrics
- `ProvenanceAgg` — Source tracking
- `Score` — Coherence scores

## Contributing

### Adding New Languages
1. Add language code to `SPACY_MODEL_MAP`
2. Install và test spaCy model
3. Update language detection logic nếu needed
4. Add tests cho new language
5. Update documentation

### Custom Scoring Algorithms
1. Extend `Score` dataclass
2. Implement scoring method trong chunker
3. Add configuration parameters
4. Test scoring accuracy
5. Update benchmarks

### Performance Optimizations
1. Profile current implementation
2. Identify bottlenecks (model loading, text processing)
3. Implement caching strategies
4. Add batch processing support
5. Benchmark improvements</content>
<parameter name="filePath">d:\Project\RAG-2\chunkers\README.MD
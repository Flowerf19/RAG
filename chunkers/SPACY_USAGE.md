# Sử dụng spaCy cho Semantic Chunking

## Tổng quan

Module `chunkers` hỗ trợ **spaCy** để cải thiện chất lượng chunking, đặc biệt cho semantic chunking (chia theo câu và đoạn văn).

## Lợi ích của spaCy

✅ **Sentence boundary detection tốt hơn**: Chính xác hơn nhiều so với regex
✅ **Hỗ trợ đa ngôn ngữ**: English, Vietnamese, và 60+ ngôn ngữ khác
✅ **Linguistic features**: Noun phrases, named entities, dependencies
✅ **Performance cao**: Code được tối ưu bằng C
✅ **Tự động fallback**: Nếu không có spaCy, tự động dùng regex

## Cài đặt

```bash
# SpaCy đã có sẵn trong requirements.txt
# Chỉ cần cài language models

# English (recommended)
python -m spacy download en_core_web_sm

# Vietnamese (optional)
python -m spacy download vi_core_news_sm

# Hoặc models lớn hơn cho accuracy cao hơn
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

## Sử dụng

### 1. Semantic Chunking với spaCy

```python
from chunkers.strategies.semantic import SemanticStrategy

# English - Sentence-based
strategy = SemanticStrategy(
    max_chunk_size=800,
    min_chunk_size=100,
    split_on="sentence",
    use_spacy=True,      # Enable spaCy
    lang='en'            # Language
)

text = """
Natural language processing is a subfield of AI. 
It deals with computers and human language. 
The goal is understanding text automatically.
"""

chunks = strategy.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
```

### 2. Paragraph-based Chunking

```python
strategy = SemanticStrategy(
    max_chunk_size=1000,
    min_chunk_size=200,
    split_on="paragraph",
    use_spacy=True,
    lang='en'
)

chunks = strategy.split_text(long_text)
```

### 3. Vietnamese Text

```python
strategy_vi = SemanticStrategy(
    max_chunk_size=800,
    min_chunk_size=100,
    split_on="sentence",
    use_spacy=True,
    lang='vi'  # Vietnamese
)

text_vi = """
Xử lý ngôn ngữ tự nhiên là một lĩnh vực của AI.
Nó liên quan đến máy tính và ngôn ngữ con người.
"""

chunks = strategy_vi.split_text(text_vi)
```

### 4. Sử dụng với Chunker

```python
from loaders import PDFLoader
from chunkers import SemanticChunker

# Load PDF
loader = PDFLoader()
pdf_doc = loader.load("document.pdf").normalize()

# Chunk với spaCy
chunker = SemanticChunker({
    "max_chunk_size": 800,
    "min_chunk_size": 100,
    "split_on": "sentence",
    "use_spacy": True,
    "lang": "en"
})

chunk_doc = chunker.chunk(pdf_doc)

print(f"Total chunks: {chunk_doc.total_chunks}")
for chunk in chunk_doc.chunks:
    print(f"{chunk.stable_id}: {chunk.get_content_preview(100)}")
```

## SpaCy Utilities

Module cung cấp `SpacyChunker` utility class:

```python
from chunkers.utils import SpacyChunker

# Initialize
spacy_chunker = SpacyChunker(lang='en')

# Split into sentences
text = "First sentence. Second sentence! Third question?"
sentences = spacy_chunker.split_into_sentences(text)
# Output: ['First sentence.', 'Second sentence!', 'Third question?']

# Merge sentences into chunks
chunks = spacy_chunker.merge_sentences_to_chunks(
    sentences,
    max_chunk_size=100,
    min_chunk_size=20
)

# Extract noun phrases
noun_phrases = spacy_chunker.extract_noun_phrases(text)

# Extract named entities
entities = spacy_chunker.extract_entities(text)
```

## Configuration trong YAML

Thêm vào `config/chunking.yaml`:

```yaml
chunking:
  semantic:
    max_chunk_size: 800
    min_chunk_size: 100
    split_on: "sentence"
    preserve_structure: true
    use_spacy: true      # Enable spaCy
    lang: "en"           # Language
```

## So sánh: spaCy vs Regex

### Ví dụ với abbreviations

**Text**: `Dr. Smith works at U.S. Inc. He has a Ph.D. degree.`

**Với regex** (sai):

- "Dr."
- "Smith works at U.S."
- "Inc."
- "He has a Ph.D."
- "degree."

**Với spaCy** (đúng):

- "Dr. Smith works at U.S. Inc."
- "He has a Ph.D. degree."

### Ví dụ với complex punctuation

**Text**: `"Are you coming?" she asked. "Yes!" he replied.`

**Với regex** (sai):

- ""Are you coming?""
- "she asked."
- ""Yes!""
- "he replied."

**Với spaCy** (đúng):

- ""Are you coming?" she asked."
- ""Yes!" he replied."

## Supported Languages

SpaCy hỗ trợ 60+ ngôn ngữ. Một số phổ biến:

- English: `en_core_web_sm`, `en_core_web_md`, `en_core_web_lg`
- Vietnamese: `vi_core_news_sm`, `vi_core_news_md`, `vi_core_news_lg`
- Chinese: `zh_core_web_sm`, `zh_core_web_md`, `zh_core_web_lg`
- Japanese: `ja_core_news_sm`, `ja_core_news_md`, `ja_core_news_lg`
- German: `de_core_news_sm`, `de_core_news_md`, `de_core_news_lg`
- French: `fr_core_news_sm`, `fr_core_news_md`, `fr_core_news_lg`
- Spanish: `es_core_news_sm`, `es_core_news_md`, `es_core_news_lg`

Xem đầy đủ: <https://spacy.io/models>

## Performance

### Model sizes

- **sm** (small): ~10-15 MB, nhanh nhất
- **md** (medium): ~40-50 MB, cân bằng
- **lg** (large): ~400-500 MB, chính xác nhất

### Benchmarks

| Model | Speed | Accuracy |
|-------|-------|----------|
| en_core_web_sm | ~500 docs/sec | Good |
| en_core_web_md | ~300 docs/sec | Better |
| en_core_web_lg | ~100 docs/sec | Best |

Khuyến nghị: Dùng **small models** cho chunking (đủ cho sentence detection).

## Troubleshooting

### Model not found

```bash
# Download model
python -m spacy download en_core_web_sm

# Verify installation
python -m spacy validate
```

### Memory issues

```python
# Tăng max_length
spacy_chunker = SpacyChunker(lang='en', max_length=2000000)
```

### Disable unused pipeline components

```python
import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
# Chỉ giữ tokenizer và sentence splitter
```

## Best Practices

1. **Dùng spaCy cho production**: Quality tốt hơn nhiều so với regex
2. **Cache models**: Reuse nlp object, không load nhiều lần
3. **Choose right model size**: Small models đủ cho chunking
4. **Process in batches**: Dùng `nlp.pipe()` cho nhiều documents
5. **Set max_length**: Tránh memory issues với large documents

## Example: Complete Pipeline

```python
from loaders import PDFLoader
from chunkers import SemanticChunker

# 1. Load PDF
loader = PDFLoader()
pdf_doc = loader.load("document.pdf")

# 2. Normalize
pdf_doc = pdf_doc.normalize()

# 3. Chunk với spaCy
chunker = SemanticChunker({
    "max_chunk_size": 800,
    "min_chunk_size": 100,
    "split_on": "sentence",
    "use_spacy": True,
    "lang": "en"
})

chunk_doc = chunker.chunk(pdf_doc)

# 4. Process chunks
for chunk in chunk_doc.chunks:
    print(f"ID: {chunk.stable_id}")
    print(f"Content: {chunk.content[:100]}...")
    print(f"Citation: {chunk.metadata.citation}")
    print()
```

## Testing

Run tests:

python chunkers/test_spacy_chunking.py

Expected output:
✓ English model (en_core_web_sm) is installed
✓ Semantic chunking working
✓ Sentence detection accurate
✓ Noun phrase extraction working

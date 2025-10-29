# RAG System – AI Coding Agent Instructions

## 🎯 System Architecture
Modular RAG pipeline: `PDF → PDFLoader → PDFDocument → HybridChunker → ChunkSet → Embedder → FAISS + BM25 + Reranking`

**Key Modules:**
- **`loaders/`** - Raw PDF extraction (text/tables) with factory patterns
- **`chunkers/`** - Multi-strategy text segmentation (semantic/rule-based/fixed-size)
- **`embedders/`** - Multi-provider embeddings (Ollama: Gemma 768-dim/BGE-M3 1024-dim, HuggingFace API/local)
- **`llm/`** - LLM integration with configuration loading, API handlers, and Streamlit UI
- **`pipeline/`** - RAG pipeline orchestration with composition architecture
- **`BM25/`** - Keyword-based search (Whoosh indexer + BM25 ranking)
- **`reranking/`** - Result reranking (BGE v2-m3 local, BGE-M3 Ollama/HF API/HF Local, Cohere API, Jina API)
- **`data/`** - FAISS indexes (.faiss), metadata maps (.pkl), summaries (.json), chunks (.txt)

**Data Flow**: `PDF → PDFLoader → PDFDocument → HybridChunker → ChunkSet → Embedder → FAISS IndexFlatIP + BM25 Index + Reranking`

---

## 🛠️ Essential Developer Workflows

### Core Pipeline Execution
```powershell
# Process all PDFs in data/pdf/ to FAISS indexes
python -m pipeline.rag_pipeline

# Test chunking functionality
python chunkers/chunk_pdf_demo.py

# Run Streamlit UI
streamlit run llm/LLM_FE.py

# Test reranking system
python reranking/test_reranker.py
```

### Chunk Caching Behavior
Pipeline maintains cache (`data/cache/processed_chunks.json`) - skips identical chunks on re-runs. Delete cache to force re-processing.

### UI Integration Pattern
```python
# Backend connector pattern for UI separation
from pipeline.backend_connector import RAGRetrievalService
retriever = RAGRetrievalService(pipeline)
results = retriever.retrieve("query", top_k=5)
context = retriever.build_context(results)
ui_items = retriever.to_ui_items(results)  # For UI display
```

---

## 🏗️ Critical Code Patterns

### 1. Factory Pattern (Universal)
Every major class uses factories for common configs:
```python
# PDF Loading
loader = PDFLoader.create_default()  # Text + tables, normalization enabled

# Embedding (Multi-provider support)
factory = EmbedderFactory()
gemma = factory.create_gemma()       # 768-dim semantic search
bge3 = factory.create_bge_m3()       # 1024-dim multilingual

# HuggingFace embedders
hf_api = factory.create_huggingface_api(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    api_token="hf_xxx"
)

# Reranking (Multi-provider support)
reranker = RerankerFactory.create_bge_local()  # BGE v2-m3 local
reranker = RerankerFactory.create_cohere(api_token="xxx")  # Cohere API
```

### 2. Constructor Injection (No Global Config)
All configuration via constructor - no YAML dependencies:
```python
# ✅ Current pattern
loader = PDFLoader(extract_tables=True, min_text_length=15)
chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
pipeline = RAGPipeline(
    output_dir="data",
    embedder_type=EmbedderType.OLLAMA,
    model_type=OllamaModelType.GEMMA
)
```

### 3. Configuration Loading Pattern
Centralized config loading with caching:
```python
from llm.config_loader import get_config
config = get_config()  # Cached singleton pattern
llm_config = config.get('llm', {})
```

### 4. Composition over Inheritance
RAGPipeline uses composition with specialized classes:
```python
class RAGPipeline:
    def __init__(self, ...):
        self.loader = PDFLoader.create_default()
        self.chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
        self.embedder = embedder_factory.create_gemma()
        self.vector_store = VectorStore(self.vectors_dir)
        self.summary_generator = SummaryGenerator(...)
        self.retriever = Retriever(self.embedder)
```

### 5. Data Model Normalization
Raw extraction separated from cleaning:
```python
pdf_doc = loader.load("doc.pdf")
pdf_doc = pdf_doc.normalize()  # Deduplication, text cleaning
```

### 6. Error Handling with Graceful Degradation
Optional dependencies handled gracefully:
```python
try:
    from BM25.ingest_manager import BM25IngestManager
    _BM25_IMPORT_ERROR = None
except Exception as exc:
    BM25IngestManager = None
    _BM25_IMPORT_ERROR = exc
```

---

## 🔧 Integration Points & Dependencies

### External Services
- **Ollama Server**: `http://localhost:11434` (required for Ollama embeddings)
- **HuggingFace API**: `https://router.huggingface.co/hf-inference/` (current endpoint)
- **Cohere/Jina APIs**: Optional reranking (require API tokens)
- **FAISS**: Vector storage and cosine similarity search (IndexFlatIP with normalized vectors)

### PDF Processing Libraries
```python
fitz (PyMuPDF)      # Primary text extraction
pdfplumber          # Table extraction fallback
camelot-py[cv]      # Advanced table parsing
```

### LLM Integration
- **Local LLM**: Ollama-based models via `llm/LLM_LOCAL.py`
- **API LLM**: External APIs via `llm/LLM_API.py` (Gemini, OpenAI, LMStudio)
- **Streamlit UI**: Chat interface via `llm/LLM_FE.py`

---

## ⚠️ Project-Specific Conventions

### Vietnamese Documentation
- Code comments and docstrings in Vietnamese
- README files in Vietnamese
- Maintain consistency for team collaboration

### Strict Single Responsibility
- **Loaders**: Raw extraction only (no chunking/normalization)
- **Chunkers**: Document → chunks only (no embedding)
- **Embedders**: Chunks → vectors only (Ollama or HuggingFace)
- **LLM**: Model integration and API handling only
- **VectorStore**: FAISS index management only
- **Retriever**: Search operations only
- **RAGPipeline**: Orchestration and composition only

### Table Handling
Tables include schema in chunk metadata:
```python
if chunk.metadata.get("block_type") == "table":
    table_payload = chunk.metadata.get("table_payload")
    headers = table_payload.header
    rows = table_payload.rows
```

### Import Path Handling
Dual import patterns for both module and direct execution:
```python
try:
    # When run as module
    from .chat_handler import build_messages
except ImportError:
    # When run directly as script
    from chat_handler import build_messages
```

### Common Pitfalls
- **Ollama Connection**: Always test connection before embedding operations
- **Dimension Mismatch**: Gemma (768) ≠ BGE-M3 (1024) - choose based on use case
- **Config Loading**: Use `get_config()` from `llm.config_loader` instead of direct YAML loading
- **Model Switching**: Use `OllamaModelSwitcher` for runtime model changes
- **Embedder Type Selection**: Choose Ollama (local) vs HuggingFace (API/local) at pipeline initialization
- **BM25 Optional**: BM25 search may not be available if dependencies missing
- **Reranking APIs**: Separate tokens required for Cohere/Jina (not HuggingFace token)

Use these patterns when extending the system or adding new modules.
# RAG System – AI Coding Agent Instructions

## 🎯 System Architecture
Modular RAG pipeline for PDF processing with hybrid retrieval (vector + keyword search):

**Pipeline Flow**: `PDF → PDFLoader → PDFDocument → HybridChunker → ChunkSet → Embedder → FAISS + BM25 → Reranking → LLM`

**Key Modules:**
- **`PDFLoaders/`** - PDF-Extract-Kit integration (layout detection, formula recognition, OCR) - ⚠️ **INCOMPLETE: loaders/ module missing**
- **`chunkers/`** - Multi-strategy text segmentation (semantic/rule-based/fixed-size/hybrid)
- **`embedders/`** - Multi-provider embeddings with factory pattern
  - Ollama: Gemma (768-dim), BGE-M3 (1024-dim)
  - HuggingFace: Local (BGE-M3 1024-dim), API (E5-Large Multilingual 1024-dim)
- **`pipeline/`** - RAG orchestration via composition (`RAGPipeline`, `VectorStore`, `Retriever`, `SummaryGenerator`)
- **`BM25/`** - Keyword search (Whoosh indexer + BM25F scoring)
- **`reranking/`** - Multi-provider reranking (BGE v2-m3 local, BGE-M3 HF API/local, Cohere, Jina)
- **`llm/`** - LLM integration (Gemini, LMStudio) with Streamlit UI
- **`data/`** - Outputs: FAISS indexes (.faiss), metadata (.pkl), summaries (.json), chunks (.txt), BM25 index

---

## 🛠️ Essential Developer Workflows

### Core Pipeline Execution
```powershell
# Process all PDFs in data/pdf/ to FAISS + BM25 indexes
python -m pipeline.rag_pipeline

# Run Streamlit UI with embedding controls
streamlit run llm/LLM_FE.py
# Access at: http://localhost:8501

# Direct embedding via UI: Use "🚀 Run Embedding" button in sidebar
# - Processes all PDFs in data/pdf/
# - Switch embedder provider in UI sidebar
# - Progress tracking with status updates
```

### Environment Setup (CRITICAL)
```powershell
# ALWAYS use fresh virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Install spaCy models (required for semantic chunking)
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
python -c "import spacy; spacy.cli.download('vi_core_news_lg')"

# Setup Ollama (if using Ollama embeddings)
ollama pull embeddinggemma:latest
ollama pull bge-m3:latest
ollama list  # Verify installation
```

### Caching Behavior (Important)
- **Chunk Cache**: `data/cache/processed_chunks.json` - skips identical chunks on re-runs
- **BM25 Cache**: `data/cache/bm25_chunk_cache.json` - stores BM25 document mappings
- **Force Re-processing**: Delete cache files or specific `.faiss`/`.pkl` files in `data/vectors/`

---

## 🏗️ Critical Code Patterns

### 1. Factory Pattern (Universal)
Every major class uses factories for common configs:
```python
# PDF Loading (⚠️ INCOMPLETE - has OCR issues)
# loader = PDFLoader.create_default()

# Embedding (Multi-provider support)
factory = EmbedderFactory()
gemma = factory.create_gemma()       # 768-dim semantic search
bge3 = factory.create_bge_m3()       # 1024-dim multilingual

# Reranking (Multi-provider support)
reranker = RerankerFactory.create_bge_local()  # BGE v2-m3 local
```

### 2. Constructor Injection (No Global Config)
All configuration via constructor - no YAML dependencies:
```python
# ✅ Current pattern
chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
pipeline = RAGPipeline(
    output_dir="data",
    embedder_type=EmbedderType.OLLAMA,
    model_type=OllamaModelType.GEMMA
)
```

### 3. Configuration Loading Pattern
Centralized config loading with caching (singleton pattern):
```python
from llm.config_loader import get_config, repo_path, paths_data_dir
config = get_config()  # Cached - loads config/app.yaml once
data_dir = paths_data_dir()  # Returns Path to data/pdf/
model_name = config['llm']['gemini']['model']  # Access nested keys
```

### 4. Composition over Inheritance
RAGPipeline uses composition with specialized classes:
```python
class RAGPipeline:
    def __init__(self, ...):
        self.chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
        self.embedder = embedder_factory.create_gemma()
        self.vector_store = VectorStore(self.vectors_dir)
        self.retriever = Retriever(self.embedder)
        # BM25 components optional (graceful degradation)
        self.bm25_ingest = BM25IngestManager(...) if available
```

### 5. Dual Import Pattern (Module vs Script Execution)
Support both module execution and direct script execution:
```python
try:
    # When run as module: python -m llm.LLM_FE
    from .chat_handler import build_messages
    from .config_loader import get_config
except ImportError:
    # When run directly: streamlit run llm/LLM_FE.py
    from chat_handler import build_messages
    from config_loader import get_config
```

---

## 🔧 Integration Points & Dependencies

### External Services
- **Ollama Server**: `http://localhost:11434` (required for Ollama embeddings)
- **HuggingFace API**: `https://router.huggingface.co/hf-inference/` (current endpoint - migrated Nov 2025)
  - E5-Large Multilingual: 1024-dim, 100+ languages, FREE
  - Requires `HF_TOKEN` env var or `.streamlit/secrets.toml`
- **Google Gemini API**: Configured via `.streamlit/secrets.toml` for LLM inference
- **LMStudio**: `http://127.0.0.1:1234/v1` for local LLM inference

### Core Libraries & Their Roles
```python
# Vector & Search
faiss-cpu           # IndexFlatIP with cosine similarity (normalized vectors)
whoosh              # BM25F keyword search indexing

# PDF Processing
PyMuPDF (fitz)      # Primary text extraction
pdfplumber          # Table extraction fallback
camelot-py          # Advanced table parsing

# NLP & Chunking
spacy               # Semantic chunking (en_core_web_sm, vi_core_news_lg)
sentence-transformers  # Local embeddings (BGE-M3)

# LLM Integration
google-generativeai # Gemini API client
openai              # LMStudio OpenAI-compatible API
streamlit           # Web UI framework
```

### FAISS Index Architecture
- **Index Type**: `IndexFlatIP` (inner product) with L2-normalized vectors → cosine similarity
- **Why Normalization**: `||a|| * ||b|| * cos(θ) = dot(a, b)` when vectors normalized
- **Storage**: `.faiss` binary + `.pkl` metadata (chunk_id, file_path, page mappings)
- **Query Flow**: `query → embed → normalize → index.search(query_vec, top_k) → results`

---

## ⚠️ Project-Specific Conventions

### Strict Single Responsibility
- **Loaders**: Raw extraction only (⚠️ **INCOMPLETE**)
- **Chunkers**: Document → chunks only (no embedding)
- **Embedders**: Chunks → vectors only (Ollama or HuggingFace)
- **LLM**: Model integration and API handling only
- **Retriever**: Search operations only
- **RAGPipeline**: Orchestration and composition only

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

### Graceful Degradation Pattern
Optional dependencies handled with try-except at module level:
```python
try:
    from BM25.ingest_manager import BM25IngestManager
    _BM25_IMPORT_ERROR = None
except Exception as exc:
    BM25IngestManager = None  # type: ignore
    _BM25_IMPORT_ERROR = exc

# Later in code: check if available before using
if BM25IngestManager is not None:
    self.bm25_ingest = BM25IngestManager(...)
```

### Common Pitfalls
- **PDFLoader Incomplete**: `loaders/` module missing - `from loaders.pdf_loader import PDFLoader` will fail
- **Fresh Venv Required**: Always use new virtual environment for reliable execution
- **Ollama Connection**: Always test `http://localhost:11434` before embedding operations
- **Dimension Mismatch**: Gemma (768) ≠ BGE-M3 (1024) - index rebuilt required when switching
- **HF API Token**: E5-Large needs `HF_TOKEN` env var or `.streamlit/secrets.toml`
- **spaCy Models**: `en_core_web_sm` and `vi_core_news_lg` must be installed manually
- **BM25 Index Locked**: Delete `MAIN_WRITELOCK` in `data/bm25_index/` if process crashes

Use these patterns when extending the system or adding new modules.
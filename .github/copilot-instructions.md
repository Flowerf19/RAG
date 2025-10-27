# RAG System – AI Coding Agent Instructions

## 🎯 System Architecture
Modular RAG pipeline with strict OOP design: `PDF → PDFLoader → PDFDocument → HybridChunker → ChunkSet → Embedder → FAISS + BM25 + Reranking`

**Key Modules:**
- **`loaders/`** - Raw PDF extraction (text/tables) with factory patterns
- **`chunkers/`** - Multi-strategy text segmentation (semantic/rule-based/fixed-size)
- **`embedders/`** - Multi-provider embeddings (Ollama: Gemma 768-dim/BGE-M3 1024-dim, HuggingFace API/local)
- **`llm/`** - LLM integration with configuration loading, API handlers, and Streamlit UI
- **`pipeline/`** - RAG pipeline orchestration with composition architecture
  - **`rag_pipeline.py`** - Main orchestrator using composition
  - **`vector_store.py`** - FAISS index management with IndexFlatIP
  - **`summary_generator.py`** - Document/batch summaries
  - **`retriever.py`** - Cosine similarity search using normalized vectors
  - **`backend_connector.py`** - UI integration and retrieval services
- **`BM25/`** - Keyword-based search (Whoosh indexer + BM25 ranking)
- **`reranking/`** - Result reranking (BGE v2-m3 local, BGE-M3 Ollama/HF API/HF Local, Cohere API, Jina API)
- **`data/`** - FAISS indexes (.faiss), metadata maps (.pkl), summaries (.json), chunks (.txt)

**Data Flow**: `PDF → PDFLoader → PDFDocument → HybridChunker → ChunkSet → Embedder → FAISS IndexFlatIP + BM25 Index + Reranking`

---

## 🛠️ Essential Developer Workflows

### Core Pipeline Execution
```powershell
# Process all PDFs in data/pdf/ to FAISS indexes
python run_pipeline.py

# Alternative: Direct module execution
python -m pipeline.rag_pipeline

# Test chunking functionality (available demo)
python chunkers/chunk_pdf_demo.py

# Test retrieval system
python test/e2e/test_rag_system.py

# Run Streamlit UI
streamlit run llm/LLM_FE.py

# Test reranking system (BGE local, Cohere API, Jina API)
python reranking/test_reranker.py
# Or use batch file: run_reranking_test.bat
```

### Testing & Validation
```powershell
# Run reranking system tests (currently implemented)
python reranking/test_reranker.py

# Run all tests with coverage (pytest configured in requirements.txt)
# Note: Test directory structure planned but not yet implemented
python -m pytest -v --cov=.  # When tests/ directory exists

# Individual component tests (planned structure)
python -m pytest tests/loaders/ -v        # PDF loading tests
python -m pytest tests/chunkers/ -v       # Chunking tests
python -m pytest tests/embedders/ -v      # Embedding tests
python -m pytest tests/pipeline/ -v       # Pipeline tests
python -m pytest tests/e2e/ -v           # End-to-end tests

# Test LLM API connections
python -c "from llm.LLM_API import LLMAPI; api = LLMAPI(); print('LLM ready')"
python -c "from llm.LLM_LOCAL import LLMLocal; llm = LLMLocal(); print('Local LLM ready')"
```

### Chunk Caching Behavior
The pipeline maintains a cache of processed chunks (`data/cache/processed_chunks.json`):
- **First run**: Generates embeddings for all chunks
- **Subsequent runs**: Skips chunks with identical content hash (no re-embedding)
- **To force re-processing**: Delete `data/cache/processed_chunks.json` before running

```powershell
# Clear cache and old indexes to force re-processing
Remove-Item "data\cache\processed_chunks.json" -ErrorAction SilentlyContinue
Remove-Item "data\vectors\*.pkl" -ErrorAction SilentlyContinue
Remove-Item "data\vectors\*.faiss" -ErrorAction SilentlyContinue
```

### LLM Integration Testing
```powershell
# Test LLM API connections
python -c "from llm.LLM_API import LLMAPI; api = LLMAPI(); print('LLM ready')"
python -c "from llm.LLM_LOCAL import LLMLocal; llm = LLMLocal(); print('Local LLM ready')"
```

### Ollama Setup (Required for Ollama embedders)
```bash
# Check available models
ollama list

# Required models for embedding
ollama pull embeddinggemma:latest
ollama pull bge-m3:latest
```

### HuggingFace Setup (Alternative to Ollama)
```bash
# Set API token (choose one method)
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# OR
$env:HUGGINGFACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Test token
python test_hf_token.py
```

**Important**: HuggingFace migrated to new Inference API endpoint:
- **New endpoint**: `https://router.huggingface.co/hf-inference/` (current)
- **Old endpoint**: `https://api-inference.huggingface.co` (deprecated Nov 2025)

---

## 🏗️ Critical Code Patterns

### 1. Factory Pattern (Universal)
**Every major class uses factories for common configs:**
```python
# PDF Loading
loader = PDFLoader.create_default()        # Text + tables, normalization enabled
loader = PDFLoader.create_text_only()      # Text only

# Embedding (Multi-provider support)
factory = EmbedderFactory()

# Ollama embedders
gemma = factory.create_gemma()             # 768-dim semantic search
bge3 = factory.create_bge_m3()             # 1024-dim multilingual

# HuggingFace embedders
hf_api = factory.create_huggingface_api(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    api_token="hf_xxx"
)
hf_local = factory.create_huggingface_local(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Reranking (Multi-provider support)
from reranking.reranker_factory import RerankerFactory
from reranking.reranker_type import RerankerType

reranker = RerankerFactory.create_bge_local()  # BGE v2-m3 local
reranker = RerankerFactory.create_bge_m3_ollama()  # BGE-M3 via Ollama
reranker = RerankerFactory.create_bge_m3_hf_api(api_token="hf_xxx")  # BGE-M3 via HF API
reranker = RerankerFactory.create_bge_m3_hf_local()  # BGE-M3 via HF local
reranker = RerankerFactory.create_cohere(api_token="xxx")  # Cohere API reranker
reranker = RerankerFactory.create_jina(api_token="xxx")  # Jina API reranker

# Fast model switching (Ollama only)
switcher = OllamaModelSwitcher()
switcher.switch_to_gemma()
embedder = switcher.current_embedder
```

### 2. Constructor Injection (No Global Config)
**All configuration via constructor - no YAML dependencies:**
```python
# ✅ Current pattern
loader = PDFLoader(extract_tables=True, min_text_length=15)
chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
pipeline = RAGPipeline(
    output_dir="data",
    embedder_type=EmbedderType.OLLAMA,
    model_type=OllamaModelType.GEMMA
)

# ❌ Deprecated (YAML auto-loading)
# loader = PDFLoader()  # Would load preprocessing.yaml
```

### 3. Configuration Loading Pattern
**Centralized config loading with caching:**
```python
from llm.config_loader import get_config

# Load app configuration (cached singleton)
config = get_config()
llm_config = config.get('llm', {})
model_settings = llm_config.get('models', {})
ui_settings = config.get('ui', {})
```

### 4. Data Model Normalization
**Raw extraction separated from cleaning:**
```python
# Load raw data
pdf_doc = loader.load("doc.pdf")

# Apply normalization when needed
pdf_doc = pdf_doc.normalize()  # Deduplication, text cleaning, etc.
```

### 5. Modular Chunking Strategies
**HybridChunker orchestrates multiple chunking approaches:**
```python
# Configure chunker with multiple strategies
chunker = HybridChunker(
    max_tokens=200,
    overlap_tokens=20,
    mode=ChunkerMode.AUTO  # Auto-selects best strategy per document section
)

# Available strategies: semantic, rule-based, fixed-size, structural-first
chunk_set = chunker.chunk(pdf_document)
```

### 6. Composition over Inheritance
**RAGPipeline uses composition with specialized classes:**
```python
from pipeline import RAGPipeline, VectorStore, SummaryGenerator, Retriever

class RAGPipeline:
    def __init__(self, ...):
        self.loader = PDFLoader.create_default()
        self.chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
        self.embedder = embedder_factory.create_gemma()
        self.vector_store = VectorStore(self.vectors_dir)      # Separate class
        self.summary_generator = SummaryGenerator(...)         # Separate class
        self.retriever = Retriever(self.embedder)              # Separate class
```

### 7. Embedder Type System
**Multiple embedding providers with unified interface:**
```python
from embedders.embedder_type import EmbedderType
from embedders.embedder_factory import EmbedderFactory

# Choose embedder type at runtime
pipeline = RAGPipeline(embedder_type=EmbedderType.HUGGINGFACE)

# Factory creates appropriate embedder
factory = EmbedderFactory()
embedder = factory.create_huggingface_api(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    api_token=hf_token
)
```

---

## 🔧 Integration Points & Dependencies

### External Services
- **Ollama Server**: `http://localhost:11434` (required for Ollama embeddings)
- **HuggingFace API**: Optional alternative to Ollama (requires API token, new endpoint: `https://router.huggingface.co/hf-inference/`)
- **Cohere API**: Optional reranking (requires API token)
- **Jina AI API**: Optional reranking (requires API token)
- **FAISS**: Vector storage and cosine similarity search (IndexFlatIP with normalized vectors)

### PDF Processing Libraries
```python
# Multiple engines for robustness
fitz (PyMuPDF)      # Primary text extraction
pdfplumber          # Table extraction fallback
camelot-py[cv]      # Advanced table parsing
```

### LLM Integration
- **Local LLM**: Ollama-based models via `llm/LLM_LOCAL.py`
- **API LLM**: External API integration via `llm/LLM_API.py` (Gemini, OpenAI, LMStudio)
- **Config Loading**: YAML-based configuration in `config/app.yaml`
- **Streamlit UI**: Chat interface via `llm/LLM_FE.py` with retrieval integration

### QA Pipeline Integration
```python
# RAGRetrievalService for UI integration
from pipeline.backend_connector import fetch_retrieval

# Retrieve relevant chunks for query
results = fetch_retrieval(
    query="search query",
    faiss_file=Path("data/vectors/doc.faiss"),
    metadata_file=Path("data/vectors/doc.pkl"),
    top_k=5
)

# Results include cosine similarity scores, text content, and page numbers
# When reranking is enabled, results are sorted by rerank score (descending)
# UI displays rerank scores when available, maintaining descending order
```

### Data Output Structure
Each processed PDF generates multiple timestamped files:
```
data/
├── vectors/
│   ├── DocumentName_vectors_20251023_143022.faiss      # Binary FAISS index (normalized vectors)
│   └── DocumentName_metadata_map_20251023_143022.pkl   # Chunk metadata (pages, provenance)
├── metadata/
│   └── DocumentName_summary_20251023_143022.json       # Human-readable document info
├── chunks/
│   └── DocumentName_chunks_20251023_143022.txt        # Raw chunk text for debugging
├── embeddings/
│   └── DocumentName_embeddings_20251023_143022.json   # Raw embeddings (optional debug)
└── cache/
    └── processed_chunks.json                           # Chunk processing cache
```

### Search & Retrieval
```python
# Direct pipeline usage for search
from pipeline import RAGPipeline
pipeline = RAGPipeline()

# Vector search against specific FAISS index
results = pipeline.search_similar(
    faiss_file=Path("data/vectors/Doc_vectors_20251023.faiss"),
    metadata_map_file=Path("data/vectors/Doc_metadata_map_20251023.pkl"),
    query_text="your search query",
    top_k=5
)

# BM25 keyword search (if available)
bm25_results = pipeline.search_bm25("keyword query", top_k=5)

# Hybrid search (vector + keyword)
hybrid_results = pipeline.hybrid_search("query", vector_weight=0.7, bm25_weight=0.3)

# Results include cosine similarity scores, text content, and page numbers
for result in results:
    print(f"Score: {result['cosine_similarity']:.4f}")
    print(f"Page: {result['page_number']}")
```

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
- **SummaryGenerator**: Summary creation and persistence only
- **Retriever**: Search operations only
- **RAGPipeline**: Orchestration and composition only

### Table Handling
```python
# Tables include schema in chunk metadata
if chunk.metadata.get("block_type") == "table":
    table_payload = chunk.metadata.get("table_payload")
    headers = table_payload.header
    rows = table_payload.rows
```

### Configuration Management
```python
# Use config_loader for centralized configuration
from llm.config_loader import get_config

config = get_config()  # Cached singleton pattern
llm_config = config.get('llm', {})
ui_config = config.get('ui', {})
```

### Testing Structure
```python
# One test class per module (when tests are implemented)
class TestPDFLoader:
    def test_initialization_default_params(self):
        """Test constructor with defaults"""

    def test_factory_create_default(self):
        """Test factory method patterns"""

# Test file organization mirrors source structure (planned)
tests/
├── loaders/test_pdf_loader.py
├── chunkers/test_chunkers.py
├── embedders/test_embedders.py
└── pipeline/test_rag_pipeline.py
```

---

## 🚨 Common Pitfalls

- **Ollama Connection**: Always test connection before embedding operations
- **Dimension Mismatch**: Gemma (768) ≠ BGE-M3 (1024) - choose based on use case
- **Memory Usage**: FAISS indexes can be large; monitor disk space in `data/vectors/`
- **PDF Encoding**: Use UTF-8 handling; some PDFs have encoding issues
- **Config Loading**: Use `get_config()` from `llm.config_loader` instead of direct YAML loading
- **Model Switching**: Use `OllamaModelSwitcher` for runtime model changes, not recreating embedders
- **Embedder Type Selection**: Choose between Ollama (local) and HuggingFace (API/local) at pipeline initialization
- **BM25 Optional**: BM25 search is optional and may not be available if dependencies are missing
- **Reranking APIs**: Cohere/Jina API tokens required for reranking (separate from HuggingFace token)
- **Reranking Display Order**: Reranking results are automatically sorted by score descending and displayed correctly in UI. The `similarity_score` field is updated to rerank score when reranking is applied.
- **Streamlit Imports**: Handle both module and direct script execution with try/except import patterns
- **Testing Status**: Only reranking tests currently implemented; main test suite planned but not yet available

Use these patterns when extending the system or adding new modules.
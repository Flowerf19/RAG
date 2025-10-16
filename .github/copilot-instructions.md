# RAG System – Copilot Development Instructions

## 🎯 System Architecture
Modular RAG pipeline with strict OOP design: `PDF → PDFLoader → PDFDocument → HybridChunker → ChunkSet → OllamaEmbedder → FAISS`

**Key Modules:**
- **`RAG_system/LOADER/`** - Raw PDF extraction (text/tables) with factory patterns
- **`RAG_system/CHUNKERS/`** - Multi-strategy text segmentation (semantic/rule-based/fixed-size)
- **`RAG_system/EMBEDDERS/`** - Ollama-only embeddings (Gemma:2048-dim, BGE-M3:1024-dim)
- **`RAG_system/pipeline/`** - RAG pipeline package with composition architecture
  - **`rag_pipeline.py`** - Main orchestrator using composition
  - **`vector_store.py`** - FAISS index management
  - **`summary_generator.py`** - Document/batch summaries
  - **`retriever.py`** - Vector similarity search using cosine similarity
- **`RAG_system/LLM/`** - Local LLM integration (Ollama + Gemini API support)
- **`data/`** - FAISS indexes (.faiss), metadata maps (.pkl), summaries (.json)
- **`test/`** - Comprehensive test suite with unit and integration tests

---

## 🛠️ Essential Developer Workflows

### Core Pipeline Execution
```powershell
# Process all PDFs in data/pdf/ to FAISS indexes (run from project root)
python RAG_system/pipeline/rag_pipeline.py

# Alternative: Use as module
cd RAG_system && python -m pipeline.rag_pipeline

# Test complete RAG query workflow
python test_query.py
```

### Testing & Validation
```powershell
# Run all tests with coverage (from project root)
python -m pytest -v --cov=RAG_system test/

# Individual module tests
python -m pytest test/loaders/test_pdf_loader.py -v
python -m pytest RAG_system/CHUNKERS/test_fixed_size_chunker.py -v

# Integration tests
python test/pipeline/test_real_pdf.py
python test/pipeline/test_pipeline_manual.py

# Manual component demos
python RAG_system/CHUNKERS/chunk_pdf_demo.py
```

### Ollama Setup (Required)
```bash
# Check available models
ollama list

# Required embedding models
ollama pull embeddinggemma:latest
ollama pull bge-m3:latest

# Required LLM models
ollama pull gemma3:1b
```

### Environment Setup
```powershell
# Install dependencies
pip install -r requirements.txt

# Verify Ollama connection
curl http://localhost:11434/api/tags

# Run from project root directory - imports depend on pythonpath
```

---

## 🏗️ Critical Code Patterns

### 1. Factory Pattern (Universal)
**Every major class uses factories for common configurations:**
```python
# PDF Loading
loader = PDFLoader.create_default()        # Text + tables, normalization enabled
loader = PDFLoader.create_text_only()      # Text extraction only
loader = PDFLoader.create_tables_only()    # Table extraction only

# Embedding (Ollama-only architecture)
switcher = OllamaModelSwitcher()
gemma_embedder = switcher.switch_to_gemma()    # 2048-dim semantic search
bge3_embedder = switcher.switch_to_bge_m3()    # 1024-dim multilingual

# Embedder factory alternative
factory = EmbedderFactory()
embedder = factory.create_ollama_nomic()
```

### 2. Constructor Injection (No Global Config)
**All configuration via constructor parameters - limited YAML usage:**
```python
# ✅ Explicit configuration pattern
loader = PDFLoader(extract_tables=True, min_text_length=15)
chunker = HybridChunker(max_tokens=200, overlap_tokens=20)

# ⚠️ YAML only for UI/LLM config (config/app.yaml)
# Core pipeline components avoid YAML dependencies
```

### 3. Data Model Normalization
**Raw extraction separated from cleaning/processing:**
```python
# Load raw data first
pdf_doc = loader.load("document.pdf")

# Apply normalization when needed
pdf_doc = pdf_doc.normalize()  # Deduplication, text cleaning, encoding fixes
```

### 4. Composition over Inheritance
**RAGPipeline orchestrates specialized components:**
```python
class RAGPipeline:
    def __init__(self, model_type=OllamaModelType.GEMMA):
        self.loader = PDFLoader.create_default()
        self.chunker = HybridChunker(max_tokens=200, overlap_tokens=20)
        self.model_switcher = OllamaModelSwitcher()
        self.embedder = self.model_switcher.switch_to_gemma()
        self.vector_store = VectorStore(self.vectors_dir)
        self.summary_generator = SummaryGenerator(...)
        self.retriever = Retriever(self.embedder)
```

### 5. Triple Output Pattern
**Each PDF processing generates exactly 3 files:**
```python
# Every processed document creates:
# 1. {doc}_vectors_{timestamp}.faiss     - Binary vector index
# 2. {doc}_metadata_map_{timestamp}.pkl  - Metadata mapping 
# 3. {doc}_summary_{timestamp}.json      - Human-readable summary
```

---

## 🔧 Integration Points & Dependencies

### External Services
- **Ollama Server**: `http://localhost:11434` (required for all embeddings + LLM)
- **Google Gemini API**: Configurable via `config/app.yaml` (optional LLM backend)
- **FAISS**: Vector storage with IndexFlatIP + cosine similarity (normalized vectors)

### PDF Processing Stack
```python
# Multiple engines for robustness
fitz (PyMuPDF)      # Primary text extraction
pdfplumber          # Table extraction fallback  
camelot-py[cv]      # Advanced table parsing
```

### LLM Integration Points
```python
# Dual LLM support
llm_client = LLMClient(model="gemma3:1b")  # Ollama local
# OR configured via config/app.yaml for Gemini API

# RAG query workflow
results = pipeline.search_similar(query, top_k=5)
response = llm_client.generate(query, context=results)
```

---

## ⚠️ Project-Specific Conventions

### Vietnamese Documentation
- Code comments and docstrings primarily in Vietnamese
- README files in Vietnamese with English technical terms
- Maintain linguistic consistency for team collaboration

### Strict Single Responsibility
- **Loaders**: PDF → PDFDocument (no chunking/embedding)
- **Chunkers**: PDFDocument → ChunkSet (no embedding/storage)
- **Embedders**: ChunkSet → embeddings (Ollama-only, no storage)
- **VectorStore**: FAISS index management only
- **RAGPipeline**: Orchestration and composition only
- **LLM**: Query processing and response generation

### Table-Aware Processing
```python
# Tables preserved in chunk metadata with full schema
if chunk.metadata.get("block_type") == "table":
    table_payload = chunk.metadata.get("table_payload")
    headers = table_payload.header
    rows = table_payload.rows
    # Table content embedded as structured text
```

### Module Import Patterns
```python
# Always import from RAG_system root
from RAG_system.LOADER.pdf_loader import PDFLoader
from RAG_system.CHUNKERS.hybrid_chunker import HybridChunker
from RAG_system.EMBEDDERS.providers.ollama import OllamaModelSwitcher
from RAG_system.pipeline.rag_pipeline import RAGPipeline

# Test files require sys.path adjustment
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### File Organization Convention
```python
# Output structure in data/
data/
├── pdf/           # Source PDFs
├── vectors/       # .faiss + .pkl files
├── metadata/      # .json summaries
└── batch_summary_{timestamp}.json  # Batch processing summary
```

---

## 🚨 Common Pitfalls & Solutions

- **Ollama Connection**: Always verify `http://localhost:11434` before pipeline execution
- **Model Dimensions**: Gemma(2048) ≠ BGE-M3(1024) - incompatible indexes
- **Memory Management**: FAISS indexes scale with document size - monitor `data/vectors/`
- **Import Paths**: Project root must be in pythonpath - use absolute imports
- **PDF Encoding**: Use UTF-8 normalization; some PDFs have encoding issues
- **Test Configuration**: `pyproject.toml` configures pythonpath=["."] for tests

### Quick Debugging Commands
```powershell
# Test Ollama connectivity
curl http://localhost:11434/api/tags

# Verify pipeline components
python -c "from RAG_system.pipeline.rag_pipeline import RAGPipeline; print('✓ Pipeline OK')"

# Check available models
ollama list
```

Use these patterns when extending functionality or adding new components to maintain architectural consistency.
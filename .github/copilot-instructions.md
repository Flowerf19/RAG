# RAG System – Copilot Development Instructions

## 🎯 System Architecture
Modular RAG pipeline with strict OOP design: `PDF → PDFLoader → PDFDocument → HybridChunker → ChunkSet → OllamaEmbedder → FAISS`

**Key Modules:**
- **`loaders/`** - Raw PDF extraction (text/tables) with factory patterns
- **`chunkers/`** - Multi-strategy text segmentation (semantic/rule-based/fixed-size)
- **`embedders/`** - Ollama-only embeddings (Gemma:2048-dim, BGE-M3:1024-dim)
- **`pipeline/`** - RAG pipeline package with composition architecture
  - **`rag_pipeline.py`** - Main orchestrator using composition
  - **`vector_store.py`** - FAISS index management
  - **`summary_generator.py`** - Document/batch summaries
  - **`retriever.py`** - Vector similarity search using cosine similarity
- **`data/`** - FAISS indexes (.faiss), metadata maps (.pkl), summaries (.json)

---

## 🛠️ Essential Developer Workflows

### Core Pipeline Execution
```powershell
# Process all PDFs in data/pdf/ to FAISS indexes
python -m pipeline.rag_pipeline

# Test search functionality
python demo_faiss_search.py
```

### Testing & Validation
```powershell
# Run all tests with coverage
python -m pytest -v --cov=loaders

# Individual module tests
python -m pytest chunkers/test_fixed_size_chunker.py -v

# Integration test
python chunkers/chunk_pdf_demo.py
```

### Ollama Setup (Required)
```bash
# Check available models
ollama list

# Required models
ollama pull embeddinggemma:latest
ollama pull bge-m3:latest
```

---

## 🏗️ Critical Code Patterns

### 1. Factory Pattern (Universal)
**Every major class uses factories for common configs:**
```python
# PDF Loading
loader = PDFLoader.create_default()        # Text + tables, normalization enabled
loader = PDFLoader.create_text_only()      # Text only

# Embedding (Ollama-only)
factory = EmbedderFactory()
gemma = factory.create_gemma()             # 2048-dim semantic search
bge3 = factory.create_bge_m3()             # 1024-dim multilingual

# Fast model switching
switcher = OllamaModelSwitcher()
switcher.switch_to_gemma()
embedder = switcher.current_embedder
```

### 2. Constructor Injection (No Global Config)
**All configuration via constructor - no YAML dependencies:**
```python
# ✅ Current pattern
loader = PDFLoader(extract_tables=True, min_text_length=15)

# ❌ Deprecated (YAML auto-loading)
# loader = PDFLoader()  # Would load preprocessing.yaml
```

### 3. Data Model Normalization
**Raw extraction separated from cleaning:**
```python
# Load raw data
pdf_doc = loader.load("doc.pdf")

# Apply normalization when needed
pdf_doc = pdf_doc.normalize()  # Deduplication, text cleaning, etc.
```

### 5. Composition over Inheritance
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

---

## 🔧 Integration Points & Dependencies

### External Services
- **Ollama Server**: `http://localhost:11434` (required for all embeddings)
- **FAISS**: Vector storage and cosine similarity search (IndexFlatIP with normalized vectors)

### PDF Processing Libraries
```python
# Multiple engines for robustness
fitz (PyMuPDF)      # Primary text extraction
pdfplumber          # Table extraction fallback
camelot-py[cv]      # Advanced table parsing
```

### Data Output Structure
Each processed PDF generates:
- **`.faiss`**: Binary vector index (compact, fast)
- **`.pkl`**: Metadata mapping (page numbers, provenance)
- **`_summary.json`**: Document info (human-readable)

---

## ⚠️ Project-Specific Conventions

### Vietnamese Documentation
- Code comments and docstrings in Vietnamese
- README files in Vietnamese
- Maintain consistency for team collaboration

### Strict Single Responsibility
- **Loaders**: Raw extraction only (no chunking/normalization)
- **Chunkers**: Document → chunks only (no embedding)
- **Embedders**: Chunks → vectors only (Ollama-only)
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

### Testing Structure
```python
# One test class per module
class TestPDFLoader:
    def test_initialization_default_params(self):
        """Test constructor with defaults"""

    def test_factory_create_default(self):
        """Test factory method patterns"""
```

---

## 🚨 Common Pitfalls

- **Ollama Connection**: Always test connection before embedding operations
- **Dimension Mismatch**: Gemma (2048) ≠ BGE-M3 (1024) - choose based on use case
- **Memory Usage**: FAISS indexes can be large; monitor disk space in `data/vectors/`
- **PDF Encoding**: Use UTF-8 handling; some PDFs have encoding issues

Use these patterns when extending the system or adding new modules.
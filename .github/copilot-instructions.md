# RAG System – AI Coding Agent Instructions

## 🎯 System Architecture
Modular RAG pipeline: `PDF → PDFLoader → PDFDocument → HybridChunker → ChunkSet → Embedder → FAISS + BM25 + Reranking`

**Key Modules:**
- **`loaders/`** - PDF extraction (text/tables) - ⚠️ **INCOMPLETE: OCR issues**
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

# Run Streamlit UI
streamlit run llm/LLM_FE.py

# Test reranking system
python reranking/test_reranker.py
```

### Environment Setup (CRITICAL)
```powershell
# ALWAYS use fresh virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Chunk Caching Behavior
Pipeline maintains cache (`data/cache/processed_chunks.json`) - skips identical chunks on re-runs. Delete cache to force re-processing.

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
Centralized config loading with caching:
```python
from llm.config_loader import get_config
config = get_config()  # Cached singleton pattern
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
```

---

## 🔧 Integration Points & Dependencies

### External Services
- **Ollama Server**: `http://localhost:11434` (required for Ollama embeddings)
- **HuggingFace API**: `https://router.huggingface.co/hf-inference/` (current endpoint)
- **FAISS**: Vector storage and cosine similarity search (IndexFlatIP with normalized vectors)

### PDF Processing Libraries
```python
fitz (PyMuPDF)      # Primary text extraction
pdfplumber          # Table extraction fallback
```

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

### Common Pitfalls
- **PDFLoader Incomplete**: OCR functionality has issues - avoid using until fixed
- **Fresh Venv Required**: Always use new virtual environment for reliable execution
- **Ollama Connection**: Always test connection before embedding operations
- **Dimension Mismatch**: Gemma (768) ≠ BGE-M3 (1024) - choose based on use case

Use these patterns when extending the system or adding new modules.
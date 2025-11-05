# RAG System – AI Coding Agent Instructions

## 🎯 System Architecture
Modular RAG pipeline for PDF processing with hybrid retrieval (vector + keyword search):

**Pipeline Flow**: `PDF → PDFProvider → PDFDocument → SemanticChunker → ChunkSet → Embedder → FAISS + BM25 → Reranking → LLM`

**Key Modules:**
- **`PDFLoaders/`** - Smart PDF loading with OCR integration via `PDFProvider`
  - Uses `PyMuPDF` (fitz) for text extraction + `PaddleOCR` for image-based PDFs
  - Auto-detection: text-based (>50 chars/page) vs image-based PDFs
  - Table extraction: `pdfplumber` + OCR enhancement (triggered when >30% empty cells)
  - Figure extraction: Groups images + OCR text extraction per figure
  - **Language mapping**: `multilingual` → `en` (PaddleOCR doesn't support `multilingual` directly)
  - **OCR Enhancement Logic**:
    - Tables: Appends `[OCR Supplement]` row when >30% cells empty
    - Figures: Extracts text via PaddleOCR and stores in `figure['text']`
  - Architecture: `PDFProvider` → `PDFDocument` (with `PageContent` list)
- **`chunkers/`** - Semantic text segmentation using spaCy + coherence scoring
  - `SemanticChunker`: spaCy sentence splitting + discourse marker analysis
  - **Multi-language support**: Auto-selects spaCy model based on language (en, vi, zh, fr, de, es, etc.)
  - **Critical**: Aggregates ALL page content (text + tables + figures) via `_aggregate_page_content()`
  - Entity overlap + lexical overlap for coherence scoring
  - Output: `ChunkSet` with `Chunk` objects (text + provenance + metadata)
- **`embedders/`** - Multi-provider embeddings with factory pattern
  - Ollama: Gemma (768-dim), BGE-M3 (1024-dim)
  - HuggingFace: Local (BGE-M3 1024-dim), API (intfloat/multilingual-e5-large 1024-dim)
  - Factory methods: `create_gemma()`, `create_bge_m3()`, `create_huggingface_local()`, `create_huggingface_api()`
- **`pipeline/`** - RAG orchestration (organized into submodules)
  - **`rag_pipeline.py`** (427 lines): Main orchestrator
  - **`backend_connector.py`** (29 lines): Backward compatibility
  - **`processing/`** - PDF and embedding processing
    - `pdf_processor.py` (120 lines): PDF → chunks
    - `embedding_processor.py` (196 lines): Chunks → embeddings
  - **`storage/`** - File I/O and vector storage
    - `file_manager.py` (181 lines): File I/O operations
    - `vector_store.py` (114 lines): FAISS operations
    - `summary_generator.py` (136 lines): Document summaries
  - **`retrieval/`** - Hybrid retrieval operations
    - `retrieval_service.py` (271 lines): Hybrid search coordination
    - `retrieval_orchestrator.py` (271 lines): `fetch_retrieval()` flow
    - `retriever.py` (112 lines): Vector search
    - `score_fusion.py` (222 lines): Score normalization
- **`query_enhancement/`** - Query expansion and preprocessing
  - **`query_processor.py`** (110 lines): Query enhancement (QEM) + embedding fusion
  - `qem_core.py`, `qem_strategy.py`, `qem_lm_client.py`: Core QEM logic
- **`BM25/`** - Keyword-based retrieval
  - **`bm25_manager.py`** (151 lines): BM25 indexing and search operations
  - `whoosh_indexer.py`, `ingest_manager.py`, `search_service.py`: Core BM25 infrastructure
# RAG — AI Coding Agent Quick Guide

Short, actionable notes to make an AI coding agent productive in this repo.

1) Big picture (one-line): PDF → PDFLoaders → SemanticChunker → ChunkSet → Embedders → FAISS + Whoosh(BM25) → Reranker → LLM/UI

2) Key directories & representative files (use these to find behavior):
   - `PDFLoaders/` — `pdf_provider.py` (OCR heuristics, page aggregation)
   - `chunkers/` — `semantic_chunker.py`, `model/chunk.py` (chunk creation + provenance)
   - `embedders/` — `embedder_factory.py`, `embedder_type.py` (provider factory; Gemma 768 vs BGE-M3 1024)
   - `pipeline/` — **ORGANIZED** into submodules:
     - `rag_pipeline.py` (427 lines) - Main orchestrator
     - `backend_connector.py` (29 lines) - Backward compat
     - `processing/` - `pdf_processor.py` (120), `embedding_processor.py` (196)
     - `storage/` - `file_manager.py` (181), `vector_store.py` (114), `summary_generator.py` (136)
     - `retrieval/` - `retrieval_service.py` (271), `retrieval_orchestrator.py` (271), `retriever.py` (112), `score_fusion.py` (222)
   - `query_enhancement/` — Query expansion module (moved to root)
     - `query_processor.py` (110 lines) - QEM + embedding fusion
     - `qem_core.py`, `qem_strategy.py`, `qem_lm_client.py` - QEM logic
   - `BM25/` — `bm25_manager.py` (151 lines), `whoosh_indexer.py`, `ingest_manager.py` (BM25 indexing & cache: `data/cache/bm25_chunk_cache.json`)
   - `reranking/` — `reranker_factory.py` (plug-in rerankers)
   - `ui/` — `app.py`, `components/` (Streamlit UI with OOP components)
   - `llm/` — `client_factory.py`, `base_client.py`, `gemini_client.py` (LLM abstraction with factory pattern)

3) Developer workflows & commands (Windows PowerShell):
   - create env & install: `python -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r requirements.txt`
   - process PDFs to vectors/BM25: `python -c "from pipeline.rag_pipeline import RAGPipeline; RAGPipeline().process_directory('data/pdf')"`
   - run UI: `streamlit run ui/app.py` (open http://localhost:8501)
   - ⚠️ deprecated: `streamlit run llm/LLM_FE.py` (use `ui/app.py` instead)

4) Project-specific conventions to follow exactly:
   - Single responsibility: loaders = extraction only; chunkers = text → chunks only; embedders = chunk → vector only; retriever = search only; UI = rendering only; LLM = model calling only.
   - Aggregation: chunkers must call `_aggregate_page_content()` to include `page.text`, `page.tables`, and `page.figures` (see `semantic_chunker.py`).
   - Factory & constructor injection: prefer factory methods (in `embedders/`, `llm/client_factory.py`) and pass configs via constructors — avoid global state.
   - OOP components: UI components (in `ui/components/`) are classes; LLM clients inherit from `BaseLLMClient`; use polymorphism and dependency injection.
   - Dual import pattern: modules support `python -m` vs direct import; use the try/except import style shown in `ui/app.py`.
   - Graceful degradation: optional providers (BM25, Ollama) are wrapped in try/except; check for None before use (see `pipeline/backend_connector.py`).

5) Integration points & gotchas:
   - Ollama (local) endpoint: http://localhost:11434 — verify before embedding.
   - Embedding dimensions mismatch: Gemma=768, BGE-M3/E5=1024 → rebuild indexes when switching providers.
   - FAISS storage: `.faiss` + `.pkl` metadata in `data/vectors/`. Whoosh index in `data/bm25_index/` (watch `MAIN_WRITELOCK`).
   - spaCy models required (install `en_core_web_sm`, `vi_core_news_lg` when needed).

6) Quick pointers for search & rerank flow (use these files to trace behavior):
   - query enhancement: `query_enhancement/query_processor.py` (uses QEM core)
   - embedding fusion: `query_processor.py` → `fuse_query_embeddings()`
   - hybrid retrieval: `pipeline/retrieval_service.py` → `retrieve_hybrid()` (calls vector + BM25, merges via `score_fusion.py`)
   - score merging: `pipeline/score_fusion.py` → `merge_vector_and_bm25()` (z-score normalization)
   - orchestration: `pipeline/retrieval_orchestrator.py` → `fetch_retrieval()` (QEM → embed → search → rerank)
   - reranking: `reranking/reranker_factory.py` returns rerankers used in `retrieval_orchestrator.py`

If any section looks incomplete or you want deeper examples (tests or small harnesses for embedders/FAISS), tell me which area to expand.

---
Updated: concise guide with file pointers and commands. Reply with which sections to expand or any missing integrations to include.
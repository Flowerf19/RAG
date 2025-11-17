 # RAG — AI Coding Agent Instructions (concise)

 This file gives focused, actionable guidance for an AI coding agent working in this repository.

 1) Big picture (one-line):
    - PDF → `PDFLoaders` → `chunkers` → `embedders` → FAISS (vectors) + Whoosh/BM25 → `reranking` → LLM/UI

 2) Core components & representative files:
    - Loaders: `PDFLoaders/pdf_provider.py` (OCR heuristics, page aggregation)
    - Chunking: `chunkers/semantic_chunker.py` (use `_aggregate_page_content()` to combine text/tables/figures)
    - Embedders: `embedders/embedder_factory.py`, `embedders/embedder_type.py` (factory pattern)
    - Orchestration: `pipeline/rag_pipeline.py` (entry points), `pipeline/processing/` (pdf → chunks → embeddings)
    - Retrieval: `pipeline/retrieval_orchestrator.py`, `pipeline/retrieval/retriever.py`, `pipeline/score_fusion.py`
    - BM25: `BM25/bm25_manager.py`, `BM25/whoosh_indexer.py` (indexes stored under `data/bm25_index/`)
    - Reranking: `reranking/reranker_factory.py` (pluggable rerankers)
    - UI: `ui/app.py` (Streamlit RAGChatApp) + `ui/components/`

 3) Essential developer workflows (bash/Linux):
    - Create venv and install: `python -m venv .venv` then `source .venv/bin/activate` and `pip install -r requirements.txt`
    - Run pipeline to process PDFs into vectors/BM25: `python -c "from pipeline.rag_pipeline import RAGPipeline; RAGPipeline().process_directory('data/pdf')"`
    - Start UI: `streamlit run ui/app.py` (default: http://localhost:8501)
    - Run tests: `pytest -q` or run specific tests like `pytest test_new_embedders.py::test_embedding_creation -q`
    - Quick single-file test: `python -c "from pipeline.rag_pipeline import RAGPipeline; p=RAGPipeline(); p.process_pdf('data/pdf/example.pdf')"`

 4) Project-specific conventions (do these exactly):
    - Single responsibility per module: loaders ⇢ extraction only; chunkers ⇢ chunk logic only; embedders ⇢ vectorization only; retriever ⇢ search only; UI ⇢ rendering only.
    - Always aggregate page assets in chunker: call `_aggregate_page_content()` in `semantic_chunker.py` to include `page.text`, `page.tables`, `page.figures`.
    - Use factory functions for embedders/LLM clients: prefer `embedders.embedder_factory` and `llm.client_factory` over direct imports.
    - Dual import compatibility: modules often support `python -m module` vs direct import — follow the try/except pattern used in `ui/app.py`.
    - Avoid global state; pass configs via constructors (see `pipeline/backend_connector.py` for pattern).

 5) Integration points & gotchas:
    - Ollama local endpoint: `http://localhost:11434` (verify before using `embedders`/`llm` that expect Ollama).
    - Embedding dimension mismatches: Gemma=768 vs BGE-M3/E5=1024 — switching embedder requires rebuilding FAISS indexes in `data/vectors/`.
    - FAISS metadata: vectors saved under `data/vectors/` as `.faiss` + `.pkl` pairs; BM25 index under `data/bm25_index/`.
    - spaCy models: some chunkers rely on language-specific models (install `en_core_web_sm`, `vi_core_news_lg` as needed).
    - Logging: call `logging.basicConfig()` before modules that instantiate loggers (see `pipeline/rag_pipeline.py`).

 6) Quick search & rerank trace (where to look when debugging):
    - Query → `query_enhancement/query_processor.py` (QEM)
    - Embedding + search → `pipeline/retrieval_orchestrator.py` → `retriever.py` (vector) + `BM25/bm25_manager.py`
    - Merge scores → `pipeline/score_fusion.py`
    - Rerank → `reranking/reranker_factory.py`

 When updating this file: keep changes minimal, reference concrete file paths above, and run `pytest -q` to validate. Reply with any missing details you want added (examples, commands, or file references).
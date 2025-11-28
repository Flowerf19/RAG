# Copilot Instructions — RAG project (concise)

This short guide helps AI coding agents be productive in this Retrieval-Augmented-Generation repository.

## Big picture
Flow: `PDFLoaders` → `chunkers` → `embedders` → Vector store (FAISS) + BM25 (Whoosh) → `query_enhancement` → `retrieval_orchestrator` → `score_fusion` → `reranking` → LLM/UI → `evaluation`.

## Key files (jump-to examples)
- Loaders: `PDFLoaders/provider/pdf_provider.py` (OCR, tables, multilingual hooks).
- Chunking: `chunkers/semantic_chunker.py` — use `_aggregate_page_content()`; it reads `page.text` and avoids duplicating `page.tables`/`page.figures`.
- Embedders: `embedders/embedder_factory.py` and `embedders/providers/*` (factory methods: `create_bge_m3()`, `create_gemma()`, etc.).
- Pipeline: `pipeline/rag_pipeline.py` (CLI/entry), `pipeline/processing/` (chunk → embed → store).
- Retrieval: `pipeline/retrieval/retrieval_orchestrator.py`, `pipeline/retrieval/retriever.py`, `pipeline/retrieval/score_fusion.py`.
- BM25: `BM25/bm25_manager.py`, `BM25/whoosh_indexer.py` (index stored under `data/bm25_index/`).
- LLM adapters: `llm/client_factory.py`, `llm/*.py` (Gemini, LMStudio, Ollama).
- UI & evaluation: `ui/app.py` (Streamlit), `evaluation/backend_dashboard/*`.

## Developer workflows (Windows PowerShell examples)
- Create env & install: `python -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r requirements.txt`
- Process PDFs (build chunks & embeddings):
	`python -c "from pipeline.rag_pipeline import RAGPipeline; RAGPipeline().process_directory('data/pdf')"`
- Run UI: `streamlit run ui/app.py`
- Run evaluation quick check:
	`python -c "from evaluation.backend_dashboard.api import BackendDashboard; BackendDashboard().evaluate_ground_truth_with_ragas(llm_provider='ollama', limit=5)"`
- Tests: `pytest -q`

## Project-specific conventions & patterns
- Factories: prefer `EmbedderFactory`, `LLMClientFactory`, `reranker_factory` for pluggable implementations.
- No global state: pass configs via constructors and `pipeline/backend_connector.py`.
- Chunking: call `_aggregate_page_content()` in `chunkers/semantic_chunker.py` to avoid duplicate content from tables/figures.
- Embedding dimensions: switching embedders (e.g. Gemma=768 vs BGE-M3/E5=1024) requires rebuilding FAISS indexes stored in `data/vectors/`.

## Integration points & important gotchas
- Ollama default API: `http://localhost:11434` — ensure Ollama daemon is running before using its embedder/LLM.
- Environment variables: `GOOGLE_API_KEY` for Gemini, `OPENAI_API_KEY` for OpenAI; LMStudio uses `base_url` in `llm/config_loader.py`.
- BM25 index path: `data/bm25_index/` (Whoosh files). Rebuild when ingest code changes.
- Evaluation: prefer `BackendDashboard._get_or_create_embedder()` to reuse cached embedders for speed.
- spaCy: install models used by `BM25/keyword_extractor.py` (e.g. `en_core_web_sm`).

## Quick troubleshooting pointers
- Query flow debug: follow `query_enhancement/query_processor.py` → `pipeline/retrieval/retrieval_orchestrator.py` → `pipeline/retrieval/retriever.py` + `BM25/bm25_manager.py` → `score_fusion` → `reranker_factory`.
- Chunker problems: inspect `chunkers/semantic_chunker.py` (sentence splitting & `_aggregate_page_content`).
- Embedding problems: check `embedders/providers/*` and `pipeline/processing/embedding_processor.py`.

If you'd like, I can (1) run `pytest -q`, (2) regenerate FAISS vectors when switching embedders, or (3) add more examples for common edits. Tell me which next.
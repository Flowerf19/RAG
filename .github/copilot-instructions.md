## Copilot Instructions — RAGFlow (Concise)

This file helps AI coding agents be productive in this Retrieval-Augmented-Generation repository.

Key goals: find the correct entry points, understand dataflow, and spot integration/config edges.

Architecture (high level)
- Flow: `PDFLoaders` → `chunkers` → `embedders` → FAISS vector store + BM25 (Whoosh) → `query_enhancement` → `retrieval_orchestrator` → `score_fusion` → `reranking` → LLM/UI → `evaluation`
- Main orchestrator: `pipeline/rag_pipeline.py` (ingestion: PDF -> chunks -> embeddings -> FAISS)
- Main retrieval flow: `pipeline/retrieval/retrieval_orchestrator.py` (QEM, embedding fusion, hybrid retrieval, optional reranking)

Entry points & jump-to files
- Ingest / CLI: `pipeline/rag_pipeline.py` — call `RAGPipeline().process_directory('data/pdf')` to produce chunks, embeddings, FAISS and BM25 indexes
- UI: `ui/app.py` — Streamlit frontend and how it invokes `fetch_retrieval` and `LLMClientFactory`
- Retrieval: `pipeline/retrieval/retrieval_orchestrator.py`, `pipeline/retrieval/retriever.py`, `pipeline/retrieval/score_fusion.py`
- Query enhancement: `query_enhancement/query_processor.py` & `query_enhancement/qem_core.py`
- Embedders/factories: `embedders/embedder_factory.py` + provider implementations under `embedders/providers/*`
- LLM/clients: `llm/client_factory.py` + clients `llm/*.py` (Gemini, LMStudio, Ollama)
- BM25: `BM25/bm25_manager.py`, `BM25/whoosh_indexer.py` (index dir: `data/bm25_index/`)

Developer workflows (Windows PowerShell commands)
- Setup env & deps:
  python -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r requirements.txt
- Run ingestion (re-generate FAISS + embeddings):
  python -c "from pipeline.rag_pipeline import RAGPipeline; RAGPipeline().process_directory('data/pdf')"
- Start UI:
  .venv\Scripts\Activate.ps1; streamlit run ui/app.py
- Pull Ollama models (if using Ollama):
  ollama pull embeddinggemma:latest; ollama pull bge-m3:latest
- Run tests:
  pytest -q

Project-specific conventions & patterns
- Factory patterns: prefer `EmbedderFactory`, `LLMClientFactory`, and `RerankerFactory` to instantiate providers without hard-coding implementation details.
- No global state: components are configured and injected via constructors (e.g., `RAGPipeline`, `RAGRetrievalService`, `QueryProcessor`). Use `pipeline/backend_connector.py` re-export when migrating older code.
- Index & output layout: ingestion writes to `data/` with subfolders `chunks/`, `embeddings/`, `vectors/` (FAISS), `bm25_index/`, `metadata/`, and `cache/`.
- Embedding dimensions matter: switching embedders (e.g., Gemma 768 vs BGE-M3/E5 1024) requires rebuilding FAISS indexes.

Integration points & environment
- Ollama: default endpoint `http://localhost:11434` — ensure Ollama daemon running for Ollama-based embedders or LLMs.
- HuggingFace: support for API or local; default token via `.streamlit/secrets` or `HF_TOKEN` env var
- Gemini/OpenAI: `GOOGLE_API_KEY`, `OPENAI_API_KEY` env vars are read by clients
- Whoosh (BM25): `data/bm25_index/` is the index directory; rebuild when ingest code or chunk formats change
- spaCy models: `en_core_web_sm` is used by BM25/keyword extractor—install when needed

Debug & troubleshooting notes (fast pointers) 
- Query & retrieval path for debugging: `query_enhancement/query_processor.py` → `pipeline/retrieval/retrieval_orchestrator.py` → `pipeline/retrieval/retriever.py` + `BM25/bm25_manager.py` → `pipeline/retrieval/score_fusion.py` → `reranking/reranker_factory.py`.
- If UI retrieval returns no results: confirm FAISS index files exist under `data/vectors/`, check embeddings shape matches FAISS dimension (run a small embedding job to verify), and confirm Whoosh index under `data/bm25_index/`.
- Optional dependencies (Whoosh, Whoosh-subpackages, spaCy, paddlenlp) may not be present in the target environment — code tries to detect and degrade gracefully.
- If `ImportError` or missing models: inspect `embedders/providers/*` or `BM25/whoosh_indexer.py` errors and set env vars or install appropriate dependencies.

Tests & CI guidance
- Use `pytest -q` to run the test suite; unit tests exist for embedders, chunker, BM25, and evaluation.
- Tests that depend on local models or APIs should be run with appropriate env variables or mocked clients (e.g., `Ollama` accessible at port 11434).

Where to start for changes
- New embedder provider: add to `embedders/providers/` and register creation helper in `EmbedderFactory`
- New LLM: add new client in `llm/` and expose in `LLMClientFactory` (use `create_from_string` for UI integration)
- Reranker: implement provider under `reranking/providers/` and add to `reranker_factory`

Notes for AI agents
- Prefer using factory functions to create instances; avoid touching global state when possible
- Keep the directory layout intact to match downstream code expectations (`data/vectors/`, `data/embeddings/`, `data/bm25_index/`)
- Use `pipeline.backend_connector` imports when modifying code that used older interfaces

If you want, I can also:
- Run `pytest -q` and fix test failures
- Regenerate FAISS indexes for a small sample PDF to verify an embedder switch
- Add more example snippets for embedding and reranker dev work

---
Generated by AI agent; please review and ask if you'd like the instructions customized further for: debugging, testing, or CI/CD integration.
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
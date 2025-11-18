 # RAG — AI Coding Agent Instructions (concise)

This file gives focused, actionable guidance for an AI coding agent working in this repository.

1) Big picture (one-line):
   - PDF → `PDFLoaders` → `chunkers` → `embedders` → FAISS (vectors) + Whoosh/BM25 → `reranking` → LLM/UI/Evaluation

2) Core components & representative files:
   - Loaders: `PDFLoaders/pdf_provider.py` (OCR heuristics, page aggregation)
   - Chunking: `chunkers/semantic_chunker.py` (use `_aggregate_page_content()` to combine text/tables/figures)
   - Embedders: `embedders/embedder_factory.py`, `embedders/embedder_type.py` (factory pattern)
   - Query Enhancement: `query_enhancement/query_processor.py` (QEM for query expansion)
   - Orchestration: `pipeline/rag_pipeline.py` (entry points), `pipeline/processing/` (pdf → chunks → embeddings)
   - Retrieval: `pipeline/retrieval_orchestrator.py`, `pipeline/retrieval/retriever.py`, `pipeline/score_fusion.py`
   - BM25: `BM25/bm25_manager.py`, `BM25/whoosh_indexer.py` (indexes stored under `data/bm25_index/`)
   - Reranking: `reranking/reranker_factory.py` (pluggable rerankers)
   - LLM: `llm/client_factory.py` (multiple providers: Ollama, Gemini, LMStudio)
   - UI: `ui/app.py` (Streamlit RAGChatApp) + `ui/components/`, `ui/dashboard/` (evaluation dashboard)
   - Evaluation: `evaluation/` (metrics, evaluators, backend_dashboard for assessing RAG performance) - **PERFORMANCE OPTIMIZED**: Use `BackendDashboard` embedder caching to avoid repeated embedder instantiation (12x speedup). **FIXED**: Removed incorrect response quality evaluation from retrieval_orchestrator - evaluation now only happens in dashboard modules with actual LLM responses and ground truth.

3) Essential developer workflows (bash/Linux; adapt for Windows e.g., activate script):
   - Create venv and install: `python -m venv .venv` then `source .venv/bin/activate` (Linux) or `.venv\Scripts\Activate.ps1` (Windows) and `pip install -r requirements.txt`
   - Install language models: `python -c "import spacy; spacy.cli.download('en_core_web_sm')"` and optionally `python -c "import spacy; spacy.cli.download('vi_core_news_lg')"`
   - Start Ollama (optional for local models): `ollama pull embeddinggemma:latest` and `ollama pull bge-m3:latest`
   - Run pipeline to process PDFs into vectors/BM25: `python -c "from pipeline.rag_pipeline import RAGPipeline; RAGPipeline().process_directory('data/pdf')"`
   - Start UI: `streamlit run ui/app.py` (default: http://localhost:8501)
   - Start evaluation dashboard: `streamlit run ui/dashboard/app.py`
   - Run ground truth evaluation: `python run_ground_truth_evaluation.py` (uses `huggingface_local` embedder by default to match vector dimensions)
   - Run tests: `pytest -q` or run specific tests like `pytest test_new_embedders.py::test_embedding_creation -q`
   - Quick single-file test: `python -c "from pipeline.rag_pipeline import RAGPipeline; p=RAGPipeline(); p.process_pdf('data/pdf/example.pdf')"`
   - Performance test: `python test_embedder_caching.py` (demonstrates 12x speedup with embedder caching)

4) Project-specific conventions (do these exactly):
   - Single responsibility per module: loaders ⇢ extraction only; chunkers ⇢ chunk logic only; embedders ⇢ vectorization only; retriever ⇢ search only; UI ⇢ rendering only; evaluation ⇢ assessment only.
   - Always aggregate page assets in chunker: call `_aggregate_page_content()` in `semantic_chunker.py` to include `page.text`, `page.tables`, `page.figures`.
   - Use factory functions for embedders/LLM clients/rerankers: prefer `embedders.embedder_factory`, `llm.client_factory`, `reranking.reranker_factory` over direct imports.
   - Dual import compatibility: modules often support `python -m module` vs direct import — follow the try/except pattern used in `ui/app.py`.
   - Avoid global state; pass configs via constructors (see `pipeline/backend_connector.py` for pattern).

5) Integration points & gotchas:
   - Ollama local endpoint: `http://localhost:11434` (verify before using `embedders`/`llm` that expect Ollama).
   - Embedding dimension mismatches: Gemma=768 vs BGE-M3/E5=1024 — switching embedder requires rebuilding FAISS indexes in `data/vectors/`.
   - FAISS metadata: vectors saved under `data/vectors/` as `.faiss` + `.pkl` pairs; BM25 index under `data/bm25_index/`.
   - LLM providers: Gemini requires `GOOGLE_API_KEY`, LMStudio local server, Ollama for local models; OpenAI requires `OPENAI_API_KEY`.
   - Reranking APIs: Some rerankers (Jina, Cohere) require API keys; check `reranking/reranker_factory.py` for requirements.
   - Evaluation dashboard: Requires ground truth data imported into database; uses `evaluation/backend_dashboard/api.py` for metrics.
   - Embedder dimension matching: Vectors created with BGE-M3 (1024 dim) - use `huggingface_local` embedder for evaluation, not `ollama` (Gemma 768 dim).
   - Performance bottleneck: Embedder instantiation in evaluation loops - always use `BackendDashboard._get_or_create_embedder()` for cached instances (12x speedup).
   - spaCy models: some chunkers rely on language-specific models (install `en_core_web_sm`, `vi_core_news_lg` as needed).
   - Logging: call `logging.basicConfig()` before modules that instantiate loggers (see `pipeline/rag_pipeline.py`).
   - Prompts: System prompts in `prompts/rag_system_prompt.txt` for LLM interactions.

6) Quick search & rerank trace (where to look when debugging):
   - Query → `query_enhancement/query_processor.py` (QEM)
   - Embedding + search → `pipeline/retrieval_orchestrator.py` → `retriever.py` (vector) + `BM25/bm25_manager.py`
   - Merge scores → `pipeline/score_fusion.py`
   - Rerank → `reranking/reranker_factory.py`
   - Evaluation → `evaluation/backend_dashboard/api.py` (recall, relevance, semantic similarity with real cosine similarity computation)

When updating this file: keep changes minimal, reference concrete file paths above, and run `pytest -q` to validate. Reply with any missing details you want added (examples, commands, or file references).
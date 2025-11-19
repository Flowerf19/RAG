 # RAG — AI Coding Agent Instructions (concise)
 # RAG — AI Coding Agent Instructions (concise)

 This file gives focused, actionable guidance for an AI coding agent working in this repository. Keep edits minimal and reference concrete files.

 1) Big picture
    - Flow: PDFLoaders (extract) → chunkers (semantic grouping) → embedders (vectors) → Vector store (FAISS) + BM25 (Whoosh) → reranking → LLM/UI → evaluation (Ragas with 4 metrics).

 2) Key files & components (quick refs)
    - Loaders: `PDFLoaders/provider/` (`pdf_provider.py`, `PDFDocument` classes)
    - Chunking: `chunkers/semantic_chunker.py` (use `_aggregate_page_content()` — aggregates `page.text` and avoids duplicate tables/figures)
    - Embedders: `embedders/embedder_factory.py`, `embedders/providers/*` (Ollama/HF implementations)
    - Orchestration: `pipeline/rag_pipeline.py` (entry point), `pipeline/processing/` (pdf → chunks → embeddings)
    - Retrieval: `pipeline/retrieval_orchestrator.py`, `pipeline/retrieval/retriever.py`, `pipeline/score_fusion.py`
    - BM25: `BM25/bm25_manager.py`, `BM25/whoosh_indexer.py` (indexes in `data/bm25_index/`)
    - Evaluation: `evaluation/backend_dashboard/ragas_evaluator.py` (4 metrics: faithfulness, context_recall, context_relevance, answer_relevancy)
    - LLMs: `llm/client_factory.py`, `llm/*_client.py` (Gemini, LMStudio, Ollama adapters)
    - UI: `ui/app.py` (Streamlit RAGChatApp) and `ui/dashboard/` (evaluation)

 3) Developer workflows (Windows PowerShell examples)
    - Create venv & install: `python -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r requirements.txt`
    - Run pipeline (process PDFs → vectors): `python -c "from pipeline.rag_pipeline import RAGPipeline; RAGPipeline().process_directory('data/pdf')"`
    - Run UI: `streamlit run ui/app.py`
    - Run evaluation dashboard: `streamlit run ui/dashboard/app.py` (with autotest: auto-import CSV + auto-run evaluation)
    - Tests: `pytest -q` (or a single test file)
    - Install Ollama models: `ollama pull gemma3:1b` and `ollama pull embeddinggemma:latest`
    - Run Ragas evaluation with Ollama: `python -c "from evaluation.backend_dashboard.api import BackendDashboard; BackendDashboard().evaluate_ground_truth_with_ragas(llm_provider='ollama', limit=5)"`

 4) Project-specific conventions (concrete)
    - Single responsibility modules: loaders ⇢ extraction, chunkers ⇢ text grouping, embedders ⇢ vectorization, retriever ⇢ search, ui ⇢ rendering, evaluation ⇢ metrics.
    - Always aggregate page assets in the chunker via `_aggregate_page_content()` (see `chunkers/semantic_chunker.py` — it intentionally uses `page.text` and skips duplicative `page.tables`/`page.figures`).
    - Use factories for pluggable implementations: `embedders.EmbedderFactory`, `llm.client_factory.LLMClientFactory`, `reranking.reranker_factory`.
    - Avoid global state: pass config objects via constructors (pattern in `pipeline/backend_connector.py`).
    - Logging: `pipeline/rag_pipeline.py` only calls `logging.basicConfig()` when run as `__main__` or if `RAG_FORCE_LOGGING` env var is set — avoid overriding host logging in libraries.

 5) Integration points & gotchas (do not overlook)
    - Ollama local endpoint: default `http://localhost:11434`. Confirm before using Ollama embedders/clients.
    - Embedding dimension mismatch: Gemma ≈ 768 dim vs BGE-M3 / E5 ≈ 1024 dim — changing embedder requires rebuilding FAISS indexes in `data/vectors/` (vectors saved as `.faiss` + `.pkl`).
    - Use `EmbedderFactory` helpers: `create_bge_m3()`, `create_gemma()`, `create_huggingface_local()` to ensure correct model/profile values.
    - LLM credentials: Gemini needs `GOOGLE_API_KEY`; OpenAI usage (if added) needs `OPENAI_API_KEY`; LMStudio may require a `base_url` for local servers.
    - Evaluation speed: prefer cached embedder creation when running many evaluations (see `evaluation/backend_dashboard` and `BackendDashboard._get_or_create_embedder()`).
    - Ragas evaluation: Now defaults to Ollama (gemma3:1b) for rate-limit-free evaluation; use `llm_provider='gemini'` to switch to Google API.

 6) Quick traces (where to look when debugging)
    - Query → `query_enhancement/query_processor.py` → `pipeline/retrieval_orchestrator.py` → `pipeline/retrieval/retriever.py` → `pipeline/score_fusion.py` → `reranking/reranker_factory.py`.
    - Chunking issues → `chunkers/semantic_chunker.py` (sentence split, `_aggregate_page_content()`) and `PDFLoaders/provider/*` (page extraction).
    - Embedding issues → `embedders/providers/*` and `pipeline/processing/embedding_processor.py`.

 7) Small examples (copy-paste)
    - Start pipeline (PowerShell):
      `python -c "from pipeline.rag_pipeline import RAGPipeline; RAGPipeline().process_directory('data/pdf')"`
    - Create a Gemma Ollama embedder:
      `from embedders.embedder_factory import EmbedderFactory; e=EmbedderFactory().create_gemma(base_url='http://localhost:11434')`

 If anything here is unclear or missing, tell me which area (chunking, embedder, retrieval, UI, or evaluation) you want expanded and I will iterate.

1) Big picture (one-line):
   - PDF → `PDFLoaders` → `chunkers` → `embedders` → FAISS (vectors) + Whoosh/BM25 → `reranking` → LLM/UI/Evaluation

2) Core components & representative files:
   - Loaders: `PDFLoaders/provider/` (multiple providers: PDFProvider, PyMuPDF4LLMProvider, SimpleTextProvider for OCR heuristics, page aggregation)
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
   - New PDF loaders: PyMuPDF4LLMProvider offers markdown extraction with better structure preservation (headings, lists, tables) compared to plain text.

6) Quick search & rerank trace (where to look when debugging):
   - Query → `query_enhancement/query_processor.py` (QEM)
   - Embedding + search → `pipeline/retrieval_orchestrator.py` → `retriever.py` (vector) + `BM25/bm25_manager.py`
   - Merge scores → `pipeline/score_fusion.py`
   - Rerank → `reranking/reranker_factory.py`
   - Evaluation → `evaluation/backend_dashboard/api.py` (recall, relevance, semantic similarity with real cosine similarity computation)

When updating this file: keep changes minimal, reference concrete file paths above, and run `pytest -q` to validate. Reply with any missing details you want added (examples, commands, or file references).
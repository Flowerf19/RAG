# RAG — AI Coding Agent Instructions (concise)

This file guides AI agents working in this RAG (Retrieval-Augmented Generation) codebase. Focus on modular, factory-based architecture for PDF processing to conversational AI.

## Big Picture Architecture
Flow: PDFLoaders (extract text/tables/figures) → chunkers (semantic grouping) → embedders (vectors) → Vector store (FAISS) + BM25 (Whoosh) → query_enhancement (expand queries) → retrieval_orchestrator (hybrid search) → score_fusion (normalize scores) → reranking (improve relevance) → LLM/UI → evaluation (Ragas with faithfulness, context_recall, context_relevancy, answer_relevancy).

## Key Components & Files
- **Loaders**: `PDFLoaders/provider/pdf_provider.py` (`PDFProvider` with OCR, table extraction, multilingual support)
- **Chunking**: `chunkers/semantic_chunker.py` (use `_aggregate_page_content()` — aggregates `page.text` only, skips duplicative `page.tables`/`page.figures`)
- **Embedders**: `embedders/embedder_factory.py` (factory pattern: `create_bge_m3()`, `create_gemma()`, `create_huggingface_local()`)
- **Pipeline**: `pipeline/rag_pipeline.py` (entry point), `pipeline/processing/` (orchestrates PDF → chunks → embeddings)
- **Query Enhancement**: `query_enhancement/query_processor.py` (enhances queries using LLMs for better retrieval)
- **Retrieval**: `pipeline/retrieval/retrieval_orchestrator.py`, `pipeline/retrieval/retriever.py`, `pipeline/retrieval/score_fusion.py`
- **Reranking**: `reranking/reranker_factory.py` (factory for reranking algorithms to improve result relevance)
- **BM25**: `BM25/bm25_manager.py`, `BM25/whoosh_indexer.py` (indexes in `data/bm25_index/`)
- **LLM**: `llm/client_factory.py` (Gemini, LMStudio, Ollama adapters)
- **UI**: `ui/app.py` (Streamlit RAGChatApp), `ui/dashboard/` (evaluation dashboard)
- **Evaluation**: `evaluation/backend_dashboard/ragas_evaluator.py` (4 Ragas metrics, defaults to Ollama gemma3:1b)

## Developer Workflows (Windows PowerShell)
- **Setup**: `python -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r requirements.txt`
- **Process PDFs**: `python -c "from pipeline.rag_pipeline import RAGPipeline; RAGPipeline().process_directory('data/pdf')"`
- **Run UI**: `streamlit run ui/app.py`
- **Run Evaluation**: `python -c "from evaluation.backend_dashboard.api import BackendDashboard; BackendDashboard().evaluate_ground_truth_with_ragas(llm_provider='ollama', limit=5)"`
- **Install Ollama Models**: `ollama pull gemma3:1b; ollama pull embeddinggemma:latest`
- **Tests**: `pytest -q`

## Project-Specific Conventions
- **Single Responsibility**: Modules handle one concern (loaders extract, chunkers group, embedders vectorize, retriever searches)
- **Page Aggregation**: Always call `_aggregate_page_content()` in `semantic_chunker.py` — uses `page.text` (includes tables/figures as text), skips `page.tables`/`page.figures` to avoid duplication
- **Factories**: Use `EmbedderFactory`, `LLMClientFactory`, `reranking.reranker_factory` for pluggable implementations
- **No Global State**: Pass configs via constructors (see `pipeline/backend_connector.py`)
- **Logging**: `pipeline/rag_pipeline.py` calls `logging.basicConfig()` only when run as `__main__` or `RAG_FORCE_LOGGING` set

## Integration Points & Gotchas
- **Ollama**: Default `http://localhost:11434`; confirm running before using Ollama embedders/LLMs
- **Embedding Dimensions**: Gemma=768 dim vs BGE-M3/E5=1024 dim — switching embedder requires rebuilding FAISS indexes in `data/vectors/`
- **LLM Credentials**: Gemini needs `GOOGLE_API_KEY`; OpenAI needs `OPENAI_API_KEY`; LMStudio uses `base_url`
- **Evaluation**: Use `BackendDashboard._get_or_create_embedder()` for cached embedders (12x speedup); defaults to Ollama for rate-limit-free evaluation
- **spaCy Models**: Install `en_core_web_sm`, `vi_core_news_lg` for chunking

## Quick Debug Traces
- **Query Flow**: User Query → `query_enhancement/query_processor.py` → `pipeline/retrieval/retrieval_orchestrator.py` → `retriever.py` (vector) + `BM25/bm25_manager.py` → `pipeline/retrieval/score_fusion.py` → `reranking/reranker_factory.py` → LLM Generation
- **Chunking Issues**: `chunkers/semantic_chunker.py` (_aggregate_page_content, sentence splitting)
- **Embedding Issues**: `embedders/providers/*`, `pipeline/processing/embedding_processor.py`

Update this file minimally, reference concrete files, and run `pytest -q` to validate.
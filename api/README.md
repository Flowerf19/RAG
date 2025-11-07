# Flask API Overview

This document captures the requirements gathered in **Phase 1** for the Flask API that will expose the existing RAG pipeline and chat capabilities so that automated test clients can orchestrate end-to-end flows.

## Key Dependencies and Responsibilities

- `pipeline.rag_pipeline.RAGPipeline` handles PDF ingestion, chunking, embedding, and vector index storage.
- `llm.chat_handler` builds chat messages from query/context/history.
- `llm.client_factory.LLMClientFactory` resolves Gemini / LM Studio clients using `config/app.yaml` defaults.
- `config/app.yaml` supplies data paths, prompt templates, and default model parameters.

The API layer must reuse these components instead of re-implementing logic. Expensive objects (FAISS indexes, BM25 managers, LLM clients) should be initialized once at startup and shared per request through lightweight service abstractions.

## Authentication (baseline requirement)

- Each request should include an `X-API-Key` header.
- Keys will be stored in `config/app.yaml` under a new `api.auth.api_keys` list (to be added later). For now we will support a single key via env var `API_KEY` or default config.
- Requests without a valid key must receive HTTP 401 with body `{ "data": null, "error": { "code": "auth_failed", "message": "Invalid API key" }, "meta": {} }`.

## Envelope Format

All responses should follow:

```jsonc
{
  "data": { /* endpoint-specific payload or null */ },
  "error": null,
  "meta": {
    "request_id": "uuid",
    "duration_ms": 42,
    "source": "rag-api"
  }
}
```

Errors set `"data": null` and populate `error` with `{ "code": "...", "message": "...", "details": {} }`.

## Sample Payloads

### POST `/api/v1/chat`

**Request**
```json
{
  "query": "Tóm tắt chương 1 của tài liệu ABC?",
  "history": [
    { "role": "user", "content": "Cho tôi biết nội dung chính của tài liệu ABC" },
    { "role": "assistant", "content": "Tài liệu ABC nói về ..." }
  ],
  "provider": "gemini",
  "top_k": 3,
  "include_sources": true
}
```

**Response**
```json
{
  "data": {
    "answer": "Chương 1 mô tả ...",
    "provider": "gemini",
    "latency_ms": 1834,
    "sources": [
      {
        "chunk_id": "abc_12",
        "score": 0.82,
        "text": "Lorem ipsum ...",
        "metadata": {
          "page": 4,
          "document_id": "abc.pdf"
        }
      }
    ]
  },
  "error": null,
  "meta": {
    "request_id": "6f9c...",
    "duration_ms": 1910,
    "source": "rag-api"
  }
}
```

### POST `/api/v1/documents/process`

Triggers batch processing for PDFs located under `data/pdf` (default) or a provided folder. Long-running jobs return an async `job_id`.

**Request**
```json
{
  "pdf_dir": "data/pdf",          // optional, defaults to config paths
  "embedder": "ollama",           // optional
  "model": "gemma",
  "async": true
}
```

**Response (async=true)**
```json
{
  "data": {
    "job_id": "job-20250211-101500-1",
    "status": "queued",
    "submitted_at": "2025-02-11T10:15:00.123Z"
  },
  "error": null,
  "meta": { "...": "..." }
}
```

**Response (async=false)**
```json
{
  "data": {
    "results": [
      {
        "file_name": "abc.pdf",
        "success": true,
        "chunks": 245,
        "embeddings": 245,
        "summary_path": "data/metadata/abc.summary.json"
      }
    ]
  },
  "error": null,
  "meta": { "...": "..." }
}
```

### GET `/api/v1/documents/{doc_id}/context`

Returns normalized chunks/metadata so that external automation can reason about the processed content.

**Response**
```json
{
  "data": {
    "document_id": "abc.pdf",
    "summary": "Tài liệu đề cập ...",
    "chunks": [
      {
        "chunk_id": "abc_0001",
        "text": "Lorem ipsum ...",
        "score": 0.64,
        "metadata": {
          "page": 1,
          "embedding_file": "data/embeddings/abc_0001.json",
          "source": "faiss"
        }
      }
    ]
  },
  "error": null,
  "meta": { "...": "..." }
}
```

### GET `/api/v1/health`

Simple readiness probe.

```json
{
  "data": {
    "status": "ok",
    "components": {
      "rag_pipeline": "ready",
      "vector_store": "ready",
      "bm25": "ready",
      "llm_clients": {
        "gemini": "configured",
        "lmstudio": "configured"
      }
    }
  },
  "error": null,
  "meta": { "...": "..." }
}
```

These payloads will guide schema definitions and contract tests in subsequent phases.

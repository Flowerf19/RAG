# Query Enhancement Module — Query Expansion for RAG

Version: LLM-powered query enhancement with multi-language support and embedding fusion.

**Short description**: The `query_enhancement/` directory contains components for expanding user queries into multiple variants using LLMs. This improves retrieval recall by generating semantically equivalent queries in different languages and formulations, then fusing their embeddings for hybrid search.

## Objectives and Scope

- **Single Responsibility**: Query expansion and embedding fusion only.
- **Multi-language Support**: Generate variants in Vietnamese, English, and other languages.
- **LLM Integration**: Use Gemini or LM Studio for intelligent query expansion.
- **Embedding Fusion**: Fuse embeddings from multiple query variants into a single mean vector to improve retrieval.
- **Fallback Handling**: Graceful degradation when LLM services are unavailable.

## High-Level Architecture

The `query_enhancement/` module consists of:

- **`query_processor.py`** — Main coordinator for query enhancement and embedding fusion.
- **`qem_core.py`** — `QueryEnhancementModule` orchestrator with configuration management.
- **`qem_lm_client.py`** — LLM client wrapper for Gemini/LM Studio backends.
- **`qem_strategy.py`** — Prompt engineering and template management.
- **`qem_utils.py`** — Utility functions for parsing, deduplication, and logging.
- **`qem_config.yaml`** — Default configuration settings.

**Data Flow**:
```
User Query → QueryProcessor → QEM Core → LLM Client → Query Variants
  ↓
Embedding Fusion → Fused Embedding → Retrieval System
```

## Key Components (Detailed)

### `QueryProcessor` (`query_processor.py`)

Main coordinator for the query enhancement workflow.

- **`enhance_query()`** — Generate query variants using the QEM.
- **`fuse_query_embeddings()`** — Generates an embedding for each query variant and fuses them into a single mean vector.
- **`create_query_processor()`** — A factory function that initializes the `QueryProcessor` and its `QueryEnhancementModule` dependency if enabled.

### `QueryEnhancementModule` (`qem_core.py`)

Core orchestrator for query expansion.

- **`enhance()`** — Main method to expand queries using an LLM.
- **Configuration management** — Loads settings from YAML with sensible defaults.
- **Backend resolution** — Auto-selects available LLM (e.g., Gemini with a fallback).
- **Error handling** — Falls back to the original query on failures.

### `QEMLLMClient` (`qem_lm_client.py`)

LLM client abstraction for multiple backends.

- **`generate_variants()`** — Calls an LLM to generate query variants.
- **Backend switching** — Supports Gemini, LM Studio, and is easily extendable.
- **API Adaptation** — Uses `LLMClientFactory` to provide a consistent interface over different LLM APIs.

### `QEMStrategy` (`qem_strategy.py`)

Prompt engineering and template management.

- **`build_prompt()`** — Constructs optimized prompts for LLM query expansion.
- **Language specification** — Dynamically includes instructions for generating variants in multiple languages.

### `QEMUtils` (`qem_utils.py`)

Utility functions for query processing.

- **`parse_llm_list()`** — Parses LLM output (JSON arrays, bulleted lists) into a structured query list.
- **`deduplicate_queries()`** — Removes duplicate variants to save processing.
- **`log_activity()`** — Writes detailed JSONL logs for debugging and analytics.
- **`clip_queries()`** — Limits the number of variants to prevent over-expansion.

## 💡 Usage Examples

### Basic Query Enhancement

```python
from query_enhancement.query_processor import QueryProcessor
from query_enhancement.qem_core import QueryEnhancementModule

# Initialize QEM (in a real app, this would be handled by the factory)
qem = QueryEnhancementModule(app_config={"llm": {"backend": "gemini"}})
processor = QueryProcessor(qem_module=qem)

# Enhance query
variants = processor.enhance_query("machine learning algorithms")
print(variants)
# Output might include: ['machine learning algorithms', 'thuật toán máy học', 'ML algorithms', 'AI methods']
```

### Embedding Fusion

```python
from embedders.embedder_factory import EmbedderFactory
from query_enhancement.query_processor import QueryProcessor
from query_enhancement.qem_core import QueryEnhancementModule

# Initialize components
embedder = EmbedderFactory.create_gemma()
qem = QueryEnhancementModule(app_config={"llm": {"backend": "gemini"}})
processor = QueryProcessor(qem_module=qem, embedder=embedder)

# 1. Enhance query to get variants
variants = processor.enhance_query("AI research")
# ['AI research', 'nghiên cứu trí tuệ nhân tạo', 'AI studies']

# 2. Fuse embeddings of all variants into a single vector
fused_embedding = processor.fuse_query_embeddings(variants)

if fused_embedding:
    print(f"Generated a single fused embedding with dimension: {len(fused_embedding)}")
else:
    print("Failed to generate embeddings.")

```

### Using the Factory

The `create_query_processor` function simplifies initialization.

```python
from query_enhancement.query_processor import create_query_processor
from embedders.embedder_factory import EmbedderFactory

embedder = EmbedderFactory.create_gemma()

# Factory handles QEM setup internally based on config
processor = create_query_processor(use_query_enhancement=True, embedder=embedder)

# Use the processor as before
variants = processor.enhance_query("What is retrieval augmented generation?")
fused_embedding = processor.fuse_query_embeddings(variants)
print(f"Fused embedding dimension: {len(fused_embedding) if fused_embedding else 'N/A'}")
```

## 🔌 API Contracts

### `QueryProcessor.enhance_query()`
- **Input**: `query_text` (str), `use_enhancement` (bool, default=True)
- **Output**: `List[str]` - A list of query variants including the original.
- **Guarantees**: Always returns at least the original query.

### `QueryProcessor.fuse_query_embeddings()`
- **Input**: `queries` (List[str])
- **Output**: `Optional[List[float]]` - A single, fused embedding vector (mean of all variant embeddings), or `None` if embedding fails.
- **Requires**: An embedder instance must be set on the `QueryProcessor`.

### `QueryEnhancementModule.enhance()`
- **Input**: `user_query` (str)
- **Output**: `List[str]` - Enhanced query variants.
- **Fallback**: Returns `[user_query]` if enhancement fails.

### `create_query_processor()`
- **Input**: `use_query_enhancement` (bool), `embedder` (any)
- **Output**: A configured `QueryProcessor` instance.

## 🧪 Testing & Validation

The code snippets in the "Usage Examples" section demonstrate the core functionality and can serve as a basis for integration tests.

**Note**: The project structure provided does not contain a runnable test suite under a `tests/` directory. The examples below are for illustrative purposes.

### Unit Test Examples
```python
# Test QEM core functionality
qem = QueryEnhancementModule(app_config={})
variants = qem.enhance("test query")
assert len(variants) >= 1
assert "test query" in variants

# Test query processor
processor = QueryProcessor(qem_module=qem)
enhanced = processor.enhance_query("test")
assert isinstance(enhanced, list)
```

## 🤝 Contributing

### Guidelines
- **Language**: Write all comments and docstrings in **English**.
- **Principles**: Follow the single responsibility principle.
- **Configuration**: Add new settings to `qem_config.yaml` and update `DEFAULT_SETTINGS` in `qem_core.py`.
- **Testing**: Add unit tests for all new logic.
- **Documentation**: Update this README and docstrings when adding or changing features.
- **LLM Backends**: To add a new backend, update `QEMLLMClient` and `qem_config.yaml`.

## Configuration

### `qem_config.yaml`
The main configuration is managed in `query_enhancement/qem_config.yaml`.

```yaml
enabled: true

# Number of variants per language. Keys are ISO 639-1 codes.
languages:
  vi: 2
  en: 2

# Overall cap on the number of queries returned (including the original).
max_total_queries: 5

# Optional: 'gemini' or 'lmstudio'. If omitted, follows the app's default_backend.
backend:

# Fallback backend if the primary one is unavailable.
fallback_backend: gemini

# Optional overrides passed to the LLM.
llm_overrides:
  temperature: 0.3
  max_tokens: 512

# Log file for QEM activity (JSON lines format).
log_path: data/logs/qem_activity.jsonl
```

### Environment Variables
- `GEMINI_API_KEY`: Required if using the Gemini backend.
- `QEM_ENABLED`: Overrides the `enabled` flag from the config file.
- `QEM_BACKEND`: Forces a specific backend (`gemini` or `lmstudio`).
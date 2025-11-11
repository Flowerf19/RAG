# Reranking Module — Result Reordering for Enhanced Relevance

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)

The reranking module provides implementations to reorder (rerank) results from retrieval systems for improved relevance accuracy. Supports both local execution (HuggingFace, Ollama) and API calls (HuggingFace Inference, Cohere, Jina).

## ✨ Key Features

- 🔄 **Provider Abstraction**: Unified interface for different reranking providers
- 🏭 **Factory Pattern**: Easy instantiation of rerankers by type
- 🔄 **Fallback Support**: Graceful degradation when models/APIs fail
- 📊 **Score-Based Ranking**: Document reordering based on relevance scores
- ⚡ **Performance Optimized**: Efficient batch processing and caching

## 🚀 Quick Start

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For local HuggingFace rerankers
pip install transformers torch
```

### Basic Usage

The module provides unified document reranking with support for multiple providers and automatic configuration management.

## 📁 Directory Contents

- `i_reranker.py` — Interface IReranker defining contract for all rerankers
- `reranker_factory.py` — Factory for creating common rerankers
- `reranker_type.py` — Enum defining reranker types
- `providers/` — Specific implementations:
  - `base_api_reranker.py` — Base class for API-based rerankers
  - `base_local_reranker.py` — Base class for local rerankers
  - `bge_m3_hf_api_reranker.py` — BGE-M3 via HuggingFace API
  - `bge_m3_hf_local_reranker.py` — BGE-M3 local via HuggingFace

## 🔌 API Contract

### Inputs/Outputs
- **Input**: Query string + list of documents to rerank
- **Output**: Reranked documents with relevance scores
- **Error Handling**: Graceful fallback when providers are unavailable

Typical data flow:
```
Query + Documents (List[str])
  -> Reranker (IReranker.rerank)
  -> RerankResult[] (sorted by score)
```

## Supported Reranker Types

The module supports the following reranker implementations:

- **`BGE_M3_HF_LOCAL`** — BGE-M3 model running locally via HuggingFace transformers
- **`BGE_M3_HF_API`** — BGE-M3 model via HuggingFace Inference API
- **`BGE_M3_OLLAMA`** — BGE-M3 model via Ollama (planned implementation)
- **`COHERE`** — Cohere's reranking API
- **`JINA`** — Jina AI's reranking API

## Architecture Pattern

**Factory Pattern Implementation**:
```
IReranker (Interface)
├── BaseAPIReranker (Abstract)
│   ├── BGE3HFAPIReranker (Concrete)
│   ├── CohereReranker (Concrete)
│   └── JinaReranker (Concrete)
└── BaseLocalReranker (Abstract)
    ├── BGE3HFLOCALReranker (Concrete)
    └── Qwen3HFLOCALReranker (Concrete - planned)
```

**Key Design Principles**:
- **Interface Segregation**: `IReranker` defines minimal contract
- **Template Method**: Base classes handle common logic
- **Factory Pattern**: `RerankerFactory` for easy instantiation
- **Fallback Handling**: Graceful degradation on errors

## ⚙️ Configuration & Setup

### Dependencies
Use virtualenv/venv and install dependencies from project's main requirements.txt. For reranking functionality, ensure installation of:

- transformers
- torch
- requests
- (additional SDKs for Cohere/Jina if used)

### Model-Specific Setup

#### BGE-M3 Local (HuggingFace)
1. Install dependencies and model downloads automatically on first use
2. Requires transformers and torch libraries

#### BGE-M3 API (HuggingFace Inference)
1. Requires requests library
2. Set up HuggingFace token with read permissions
3. Configure HF_TOKEN environment variable

#### Cohere API
1. Install Cohere SDK
2. Configure COHERE_API_KEY environment variable

#### Jina API
1. Uses requests library
2. Configure JINA_API_KEY environment variable

## 🚀 Khởi động nhanh — ví dụ sử dụng

Ví dụ cơ bản dùng RerankerFactory:

```python
from reranking.reranker_factory import RerankerFactory
from reranking.reranker_type import RerankerType

# 1) HF local (BGE-M3)
reranker_local = RerankerFactory.create(
    reranker_type=RerankerType.BGE_M3_HF_LOCAL,
    model_name="BAAI/bge-reranker-v2-m3",
    device="cpu"
)

# 2) HF API (sử dụng HF token)
hf_token = "hf_xxx"
reranker_api = RerankerFactory.create(
    reranker_type=RerankerType.BGE_M3_HF_API,
    api_token=hf_token,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3) Cohere (API)
# reranker_cohere = RerankerFactory.create(RerankerType.COHERE, api_token="cohere-key")

# Call rerank
query = "Tác dụng của gradient descent trong machine learning"
documents = [
    "Gradient descent là thuật toán tối ưu...",
    "Support vector machines (SVM) là...",
    "Trong tối ưu học, learning rate quyết định..."
]

## 💡 Usage Examples

### Basic Usage

```python
from reranking.reranker_factory import RerankerFactory

# Create local reranker
reranker = RerankerFactory.create(RerankerType.BGE_M3_HF_LOCAL)

query = "What is machine learning?"
documents = ["ML is...", "AI includes...", "Deep learning..."]

results = reranker.rerank(query=query, documents=documents, top_k=3)
for r in results:
    print(r.index, r.score, r.document[:120])
```

### Integration with Pipeline

The reranking module integrates into RAG pipeline after retrieval to reorder results by relevance.

- Pipeline typically calls `rerank` on document lists from retrievers
- Fallback mechanism returns original order with score 0.0 on errors

## 🔌 API Contract

### Inputs/Outputs
- **Input**: query (str), documents (List[str]), top_k (int)
- **Output**: List[RerankResult] with index, score, document, and metadata

## ⚠️ Operational Notes

### Edge Cases
- Model loading failures: Returns fallback with score 0.0
- API errors: Logs error and returns fallback
- Empty documents: Returns empty list
- top_k > document count: Returns all documents

### Logging & Debugging
- Module logs at info/error levels
- For debugging: Call test_connection() and check logs

## 🧪 Testing & Validation

### Unit Tests
Test individual reranker components and integration scenarios.

### Manual Testing
Verify reranker functionality with sample queries and documents.

## 🔧 Troubleshooting

### Common Issues

**Model Loading Failures (Local):**
Check transformers/torch versions and use CPU if encountering memory issues.

**API Authentication Errors:**
Verify API tokens and permissions for HuggingFace, Cohere, or Jina services.

**Unexpected Response Formats:**
System falls back to score 0.0 for problematic responses.

## � Contributing

### Adding New Providers
When adding new reranking providers, extend BaseLocalReranker or BaseAPIReranker classes.
Include comprehensive tests and update documentation.

## 📚 Technical Reference

### Key Implementation Files
- `providers/base_local_reranker.py` — Base class for local rerankers
- `providers/base_api_reranker.py` — Base class for API rerankers
- `providers/bge_m3_hf_local_reranker.py` — BGE-M3 local implementation
- `providers/bge_m3_hf_api_reranker.py` — BGE-M3 API implementation

### Integration Points
- Pipeline: `pipeline/rag_pipeline.py`
- Configuration: `config/app.yaml`



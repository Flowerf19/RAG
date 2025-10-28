# Reranking Module - README

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)

Reranking module cung cấp các implementation để sắp xếp (rerank) kết quả trả về từ các bộ truy vấn (retrieval) theo mức độ liên quan chính xác hơn. Hỗ trợ cả chạy local (HuggingFace, Ollama) và gọi API (HuggingFace Inference, Cohere, Jina).

## Mục tiêu và phạm vi

- Tách biệt logic reranking khỏi retrieval/generation.
- Cung cấp interface chung (IReranker) dễ thay thế/khai thác.
- Hỗ trợ fallback an toàn khi model/API gặp lỗi.
- Tập trung vào việc rerank danh sách documents dựa trên query, trả về kết quả với score liên quan.

## Kiến trúc tổng quan

Thư mục `reranking/` gồm các phần chính:

- `i_reranker.py` — Interface IReranker định nghĩa hợp đồng cho tất cả reranker.
- `reranker_factory.py` — Factory để tạo nhanh các reranker phổ biến.
- `reranker_type.py` — Enum định nghĩa các loại reranker.
- `providers/` — Các implementation cụ thể:
  - `base_api_reranker.py` — Base class cho API-based reranker.
  - `base_local_reranker.py` — Base class cho local reranker.
  - `bge_m3_hf_api_reranker.py` — BGE-M3 via HuggingFace API.
  - `bge_m3_hf_local_reranker.py` — BGE-M3 local via HuggingFace.

Luồng dữ liệu điển hình:

```text
Query + Documents (List[str])
  -> Reranker (IReranker.rerank)
  -> RerankResult[] (sorted by score)
```

## Các module chính (chi tiết)

### i_reranker.py

- Mục đích: Định nghĩa interface chung cho tất cả reranker.
- Tính năng:
  - `profile` property: Trả về RerankerProfile (model_id, provider, max_lengths, is_local).
  - `rerank(query: str, documents: List[str], top_k: int = 10) -> List[RerankResult]`: Thực hiện rerank.
  - `test_connection() -> bool`: Kiểm tra kết nối/model.

### reranker_factory.py

- Mục đích: Factory để tạo reranker dễ dàng.
- Tính năng:
  - `create(reranker_type, api_token=None, model_name=None, device="cpu")`: Tạo reranker dựa trên type.

### providers/

- `base_api_reranker.py`: Base class cho reranker sử dụng API. Cung cấp `_call_api`, `_initialize_profile`.
- `base_local_reranker.py`: Base class cho reranker local. Cung cấp `_load_model`, `_compute_scores`.
- `bge_m3_hf_api_reranker.py`: Implementation cho BGE-M3 via HF API.
- `bge_m3_hf_local_reranker.py`: Implementation cho BGE-M3 local.

## 🔧 Cài đặt và thiết lập model

Sử dụng virtualenv / venv và cài dependencies trong requirements.txt của project chính. Để chạy reranking, đảm bảo cài:

- transformers
- torch
- requests
- (thêm các SDK nếu dùng Cohere/Jina)

### Cài đặt model cụ thể

#### BGE-M3 Local (HuggingFace)

1. Cài đặt dependencies:
   ```bash
   pip install transformers torch
   ```

2. Download model:
   ```python
   from transformers import AutoModelForSequenceClassification, AutoTokenizer

   model_name = "BAAI/bge-reranker-v2-m3"
   model = AutoModelForSequenceClassification.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   ```

   Model sẽ được download tự động khi khởi tạo reranker.

#### BGE-M3 API (HuggingFace Inference)

1. Cài đặt dependencies:
   ```bash
   pip install requests
   ```

2. Thiết lập token: Đăng ký tại HuggingFace, tạo token với quyền Read.

3. Environment variable:
   ```bash
   export HF_TOKEN="your_hf_token_here"
   ```

#### Cohere API

1. Cài đặt SDK:
   ```bash
   pip install cohere
   ```

2. Thiết lập API key:
   ```bash
   export COHERE_API_KEY="your_cohere_key"
   ```

#### Jina API

1. Cài đặt nếu cần (thường dùng requests).

2. Thiết lập API key:
   ```bash
   export JINA_API_KEY="your_jina_key"
   ```

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

results = reranker_local.rerank(query=query, documents=documents, top_k=3)
for r in results:
    print(r.index, r.score, r.document[:120])
```

## Hành vi tích hợp với pipeline

Module reranking được thiết kế để tích hợp vào pipeline RAG, sau retrieval để rerank kết quả.

- Pipeline thường gọi `rerank` trên danh sách documents từ retriever.
- Cơ chế fallback: Nếu lỗi, trả về thứ tự gốc với score 0.0.

Ví dụ trong pipeline:

```python
from reranking.reranker_factory import RerankerFactory

reranker = RerankerFactory.create(RerankerType.BGE_M3_HF_LOCAL)
# Sau retrieval
retrieved_docs = ["doc1", "doc2", "doc3"]
reranked = reranker.rerank(query, retrieved_docs)
```

## Contract (tóm tắt API / dữ liệu)

- Input: query (str), documents (List[str]), top_k (int)
- Output: List[RerankResult]
  - RerankResult: index (int), score (float), document (str), metadata (dict)

## Edge cases và cách xử lý

- Model không load được: Fallback trả score 0.0.
- API lỗi: Log lỗi, trả fallback.
- Documents rỗng: Trả list rỗng.
- top_k > len(documents): Trả tất cả.

## Logging & Debugging

- Module ghi log ở mức info/error.
- Để debug: Gọi test_connection(), kiểm tra log.

## Kiểm thử

- Repository có pytest. Chạy:
  ```bash
  python -m pytest tests/reranking -v
  ```

- Ví dụ unit test:

```python
def test_reranker_interface_basic():
    from reranking.reranker_factory import RerankerFactory
    from reranking.reranker_type import RerankerType

    reranker = RerankerFactory.create(RerankerType.BGE_M3_HF_LOCAL, model_name="BAAI/bge-reranker-v2-m3", device="cpu")
    assert reranker.test_connection()
    docs = ["a", "b", "c"]
    res = reranker.rerank("test query", docs, top_k=2)
    assert isinstance(res, list)
    assert all(hasattr(r, "score") for r in res)
```

## 🚨 Troubleshooting

- Model không load được (local):
  - Kiểm tra version transformers/torch.
  - Nếu OOM, dùng device="cpu".

- HF API lỗi 403/401:
  - Kiểm tra token.

- Response format unexpected: Fallback scores 0.0.

## 🧩 Mở rộng / Contribution

- Thêm provider mới: Kế thừa BaseLocalReranker hoặc BaseAPIReranker.
- Viết tests và cập nhật docs.

## Tài liệu tham chiếu

- Pipeline: `pipeline/rag_pipeline.py`
- Config: `config/app.yaml`

## Ghi chú triển khai

- README mô tả theo conventions. Kiểm tra code nếu khác.

## Chi tiết kỹ thuật theo file

### providers/base_local_reranker.py

- Base class cho local reranker.
- Methods: _load_model, _compute_scores.

### providers/base_api_reranker.py

- Base class cho API reranker.
- Methods: _call_api, _initialize_profile.

### providers/bge_m3_hf_local_reranker.py

- Implementation BGE-M3 local.
- Sử dụng transformers để load model và compute scores.

### providers/bge_m3_hf_api_reranker.py

- Implementation BGE-M3 API.
- Gọi HF Inference API.

---

Nếu cần, tôi có thể đọc code để đồng bộ chi tiết.

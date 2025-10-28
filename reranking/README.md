# Reranking Module - README

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)

Reranking module cung cấp các implementation để sắp xếp (rerank) kết quả trả về từ các bộ truy vấn (retrieval) theo mức độ liên quan chính xác hơn. Hỗ trợ cả chạy local (HuggingFace, Ollama) và gọi API (HuggingFace Inference, Cohere, Jina).

Mục tiêu:
- Tách biệt logic reranking khỏi retrieval/generation.
- Cung cấp interface chung (IReranker) dễ thay thế/khai thác.
- Hỗ trợ fallback an toàn khi model/API gặp lỗi.

---

## ✨ Tính năng chính

- Unified IReranker interface cho mọi provider (local/API).
- Các factory helper để khởi tạo reranker phổ biến (BGE-M3 local, BGE-M3 HF API, Cohere, Jina, Ollama).
- Base classes giúp viết provider mới nhanh chóng:
  - BaseLocalReranker: local HF/Ollama models
  - BaseAPIReranker: wrapper cho các API-based providers
- RerankResult chứa index gốc, score, document và metadata.
- Cơ chế graceful degradation: khi lỗi xảy ra sẽ trả về thứ tự gốc với score 0.0.

---

## 📦 Các Provider hiện có

- BGE-M3 HuggingFace (local) — BAAI/bge-reranker-v2-m3
- BGE-M3 HuggingFace (API) — default fallback sử dụng sentence-transformers/all-MiniLM-L6-v2 (do HF inference public không luôn hỗ trợ reranker models trực tiếp)
- BGE-M3 Ollama (kế hoạch / factory hỗ trợ)
- Cohere (API) — (nếu triển khai provider)
- Jina (API) — (nếu triển khai provider)

---

## 🔧 Cài đặt (tương tự project)

Sử dụng virtualenv / venv và cài dependencies trong requirements.txt của project chính. Nếu chỉ cần chạy unit tests module reranking, đảm bảo cài:
- transformers
- torch
- requests
- (thêm các SDK nếu dùng Cohere/Jina)

---

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

---

## API / Interface

- IReranker
  - property profile -> RerankerProfile (model_id, provider, max lengths, is_local)
  - rerank(query: str, documents: List[str], top_k: int = 10) -> List[RerankResult]
  - test_connection() -> bool

- RerankerFactory (tạo nhanh các reranker phổ biến)
  - create(reranker_type, api_token=None, model_name=None, device="cpu")

- RerankResult
  - index: int (index gốc trong danh sách documents)
  - score: float (score tính toán)
  - document: str
  - metadata: dict (tuỳ chọn)

---

## Cấu hình model & lưu ý

- BGE-M3 Local: cần download model HF (BAAI/bge-reranker-v2-m3). Yêu cầu có GPU nếu muốn tăng tốc.
- HF Inference API: sử dụng token từ HuggingFace. Default endpoint trong code hiện tại là `https://api-inference.huggingface.co` (chú ý: HF có router endpoint mới — kiểm tra README chính để cập nhật).
- API-based providers yêu cầu truyền api_token khi khởi tạo qua factory.

Environment variables / secrets:
- HF_TOKEN / HUGGINGFACE_TOKEN — cho HF API
- COHERE_API_KEY — (nếu dùng Cohere)
- JINA_API_KEY — (nếu dùng Jina)

---

## Kiểm thử & debug

- test_connection() — mỗi implementation cung cấp method để kiểm tra kết nối / model loaded.
- Khi lỗi xảy ra trong _call_api hoặc _compute_scores, module sẽ log lỗi và trả về kết quả fallback (score=0.0 theo thứ tự gốc).
- Để debug local HF model:
  - Kiểm tra cài torch và phiên bản transformers phù hợp.
  - Nếu dùng GPU, set device="cuda" khi khởi tạo.

---

## Ví dụ unit test (quick)

```python
def test_reranker_interface_basic():
    from reranking.reranker_factory import RerankerFactory
    from reranking.reranker_type import RerankerType

    # Nếu không có HF token, test local
    reranker = RerankerFactory.create(RerankerType.BGE_M3_HF_LOCAL, model_name="BAAI/bge-reranker-v2-m3", device="cpu")
    assert reranker.test_connection() is True or isinstance(reranker.test_connection(), bool)
    docs = ["a", "b", "c"]
    res = reranker.rerank("test query", docs, top_k=2)
    assert isinstance(res, list)
    assert all(hasattr(r, "score") for r in res)
```

---

## 🚨 Troubleshooting

- Model không load được (local):
  - Kiểm tra log lỗi (version transformers / torch).
  - Nếu OOM trên GPU, thử device="cpu" hoặc giảm batch/process size.

- HF API trả về lỗi 403/401:
  - Kiểm tra token hợp lệ, quyền `Read`.
  - Đảm bảo endpoint và header Authorization đúng.

- Response format unexpected (HF sentence-transformers):
  - Một số model trả về cấu trúc JSON khác; BaseAPIReranker có fallback đưa ra scores 0.0.

---

## 🧩 Mở rộng / Contribution

- Thêm provider mới:
  - Viết class kế thừa BaseLocalReranker hoặc BaseAPIReranker.
  - Implement _load_model/_compute_scores hoặc _initialize_profile/_call_api.
  - Đăng ký factory method trong RerankerFactory và cập nhật RerankerType nếu cần.
- Viết tests cho provider mới và cập nhật docs.

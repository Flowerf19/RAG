# RAG Pipeline - FAISS + Ollama

Hệ thống RAG (Retrieval-Augmented Generation) modular, xử lý PDF thành FAISS vector index cho tìm kiếm ngữ nghĩa siêu nhanh.

## 🚀 Khởi động nhanh

### Yêu cầu

- Python >= 3.13
- Ollama server (`http://localhost:11434`)
- Model Ollama: `embeddinggemma:latest`, `bge-m3:latest`

### Cài đặt

```bash
pip install -r requirements.txt
```

### Chạy pipeline

```powershell
python run_pipeline.py
```

- Tất cả PDF trong `data/pdf/` sẽ được xử lý thành FAISS index, embedding, metadata.

## 📁 Cấu trúc thư mục

pipeline/         # Orchestrator, FAISS, summary, retriever
loaders/          # PDF loader, table/text extraction
chunkers/         # Chunking: semantic, rule-based, fixed-size
embedders/        # Ollama embedding providers
llm/              # LLM API, config
requirements.txt  # Python dependencies
data/
  pdf/            # Nguồn PDF
  vectors/        # FAISS index (.faiss, .pkl)
  metadata/       # Document summaries (.json)
```

## 🔍 Sử dụng trong code

```python
from pipeline import RAGPipeline
pipeline = RAGPipeline()
results = pipeline.search_similar(
    faiss_file=Path("data/vectors/Doc_vectors.faiss"),
    metadata_map_file=Path("data/vectors/Doc_metadata_map.pkl"),
    query_text="your search query",
    top_k=5
)
```

## 🏗️ Kiến trúc

1. **PDF Loading**: Trích xuất text, bảng, metadata
2. **Chunking**: Chia nhỏ tài liệu theo ngữ nghĩa
3. **Embedding**: Chuyển chunk thành vector bằng Ollama
4. **FAISS Indexing**: Lưu vector cho tìm kiếm siêu nhanh

## 📊 Output

- `.faiss`: FAISS vector index
- `.pkl`: Metadata mapping
- `_summary.json`: Thông tin tài liệu

## 🎯 Ollama Embedders

```python
from embedders.embedder_factory import EmbedderFactory
factory = EmbedderFactory()
gemma = factory.create_gemma() # 768-dim
bge3 = factory.create_bge_m3() # 1024-dim
```

## 🔧 Troubleshooting

- Ollama không kết nối: kiểm tra server, model
- Model chưa có: `ollama pull embeddinggemma:latest`
- Test embedding: dùng `embedder.test_connection()`

## ✅ Sản phẩm

- FAISS lưu trữ nhỏ gọn, tìm kiếm nhanh
- Thiết kế module, dễ mở rộng
- Xử lý lỗi tốt, có fallback
- Embedding đã chuẩn hóa cho cosine similarity

---
**Xem chi tiết cải tiến, lỗi, và hướng dẫn tại IMPROVEMENTS_LOG.md**

# Pipeline Module - RAG System Architecture

## 🎯 Tổng Quan

Thư mục `pipeline/` chứa các thành phần cốt lõi của hệ thống RAG (Retrieval-Augmented Generation), chịu trách nhiệm điều phối toàn bộ quy trình xử lý từ PDF đến vector storage và retrieval.

## 🏗️ Kiến Trúc Pipeline

```mermaid
PDF Document → PDFLoader → PDFDocument → HybridChunker → ChunkSet → OllamaEmbedder → FAISS Index
```

### Data Flow

1. **PDFLoader**: Trích xuất nội dung PDF (văn bản và bảng)
2. **HybridChunker**: Phân đoạn tài liệu thành các chunk
3. **OllamaEmbedder**: Chuyển đổi chunks thành vector embeddings
4. **VectorStore**: Lưu trữ vector vào FAISS index
5. **Retriever**: Tìm kiếm tương tự dựa trên cosine similarity
6. **SummaryGenerator**: Tạo tóm tắt tài liệu và báo cáo xử lý

## 📁 Cấu Trúc Thư Mục

```yaml
pipeline/
├── rag_pipeline.py         # Orchestrator chính
├── vector_store.py         # Quản lý FAISS index
├── retriever.py            # Tìm kiếm vector similarity
├── summary_generator.py    # Tạo tóm tắt tài liệu
└── backend_connector.py    # Kết nối với backend (nếu có)
```

## 🧩 Các Thành Phần Chính

### 1. RAGPipeline (`rag_pipeline.py`)

**Trách nhiệm**: Điều phối toàn bộ quy trình xử lý PDF → Vector Storage

**Chức năng chính**:

- Xử lý hàng loạt các file PDF
- Tạo embeddings sử dụng Ollama (Gemma/BGE-M3)
- Lưu trữ vector vào FAISS index
- Tạo tóm tắt tài liệu và metadata
- Quản lý cache để tránh xử lý trùng lặp

**Khởi tạo**:

```python
pipeline = RAGPipeline(
    output_dir="data",
    pdf_dir="data/pdf",
    model_type=OllamaModelType.GEMMA  # hoặc BGE_M3
)
```

### 2. VectorStore (`vector_store.py`)

**Trách nhiệm**: Quản lý FAISS index và metadata

**Chức năng chính**:

- Tạo FAISS index từ dữ liệu embeddings
- Lưu trữ và tải FAISS index
- Quản lý metadata map cho từng chunk

### 3. Retriever (`retriever.py`)

**Trách nhiệm**: Tìm kiếm vector similarity

**Chức năng chính**:

- Tìm kiếm tương tự dựa trên cosine similarity
- Tải FAISS index và metadata
- Trả về kết quả có điểm số similarity

### 4. SummaryGenerator (`summary_generator.py`)

**Trách nhiệm**: Tạo tóm tắt tài liệu

**Chức năng chính**:

- Tạo tóm tắt cho từng tài liệu
- Tạo báo cáo xử lý hàng loạt
- Lưu trữ tóm tắt dưới dạng JSON

## 🚀 Cách Sử Dụng

### Chạy Pipeline Chính

```bash
python run_pipeline.py
```

### Sử Dụng Trực Tiếp

```python
from pipeline import RAGPipeline

# Khởi tạo pipeline
pipeline = RAGPipeline()

# Xử lý tất cả PDF trong thư mục data/pdf/
pipeline.process_all_pdfs()

# Tìm kiếm
results = pipeline.search_similar(
    faiss_file=Path("data/vectors/document_vectors.faiss"),
    metadata_map_file=Path("data/vectors/document_metadata.pkl"),
    query_text="nội dung tìm kiếm",
    top_k=5
)
```

## ⚙️ Cấu Hình

Pipeline sử dụng các thư mục mặc định:

- `data/pdf/` - Thư mục chứa PDF đầu vào
- `data/vectors/` - Lưu trữ FAISS indexes
- `data/metadata/` - Lưu trữ tóm tắt tài liệu
- `data/chunks/` - Lưu trữ chunks (debug)
- `data/cache/` - Cache xử lý để tránh trùng lặp

## 🧪 Testing

```bash
# Chạy tests cho pipeline
python -m pytest test/pipeline/ -v

# Test pipeline thủ công
python test/pipeline/test_pipeline_manual.py

# Test với PDF thực tế
python test/pipeline/test_real_pdf.py
```

## 📦 Tích Hợp BM25 (Tùy Chọn)

Pipeline hỗ trợ tích hợp BM25 search thông qua module BM25:

- `BM25IngestManager` - Quản lý ingestion
- `WhooshIndexer` - Tạo Whoosh index
- `BM25SearchService` - Dịch vụ tìm kiếm BM25

## 🔄 Composition Pattern

Pipeline sử dụng pattern Composition thay vì Inheritance:

- Mỗi class có trách nhiệm đơn lẻ (Single Responsibility)
- Dễ dàng mở rộng và bảo trì
- Tách biệt rõ ràng giữa xử lý và lưu trữ

## 📚 Xem Thêm

- [README tổng quan hệ thống](../README.md)
- [Hướng dẫn chạy pipeline](../README_RUN.md)
- [Cấu trúc dự án chi tiết](../README_STRUCTURE.md)

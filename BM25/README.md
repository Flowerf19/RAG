# BM25 Module Overview

Thư mục `BM25/` chứa toàn bộ thành phần phục vụ luồng tìm kiếm BM25 trong dự án:

- `ingest_manager.py` – điều phối ingest, nhận các chunk đã xử lý từ pipeline, trích keyword và upsert vào chỉ mục.
- `keyword_extractor.py` – tiện ích dựa trên spaCy để chuẩn hóa văn bản EN/VI và trích keyword.
- `search_service.py` – dịch vụ truy vấn, thực thi BM25 và chuẩn hóa điểm số (z-score).
- `__init__.py` – bootstrap package, export các class chính (`BM25IngestManager`, `BM25SearchService`, …).

> Lưu ý: cần triển khai thêm backend Whoosh (`whoosh_indexer.py`) và tích hợp vào pipeline để kích hoạt đầy đủ chức năng BM25.

### TODO
1. Xây dựng lớp Whoosh indexer hiện thực các giao diện `upsert_documents`, `delete_documents`, `search`.
2. Liên kết `RAGPipeline` với `BM25IngestManager` (ingest chunk sau khi tạo) và `BM25SearchService` (retrieval).
3. Cập nhật tài liệu chính (`README.md`, `RAG_QUICKSTART.md`) và cấu hình (ví dụ `config/app.yaml`) để mô tả cách bật BM25/hybrid.

# Tổng quan luồng xử lý RAG

Tài liệu này mô tả luồng xử lý tổng thể của hệ thống `ChatBot/RAG`, bao gồm hai pha chính: ingest dữ liệu (offline) và phục vụ truy vấn (online). Cuối tài liệu là phân tích chi tiết cho mô-đun viết lại truy vấn (Query Enhancement Module) và tầng rerank.

## 1. Đường ingest tài liệu (offline)

1. **Nạp & chuẩn hóa PDF**  
   - `RAGPipeline` khởi tạo `PDFLoader` mặc định (`pipeline/rag_pipeline.py:59`).  
   - `PDFLoader` dựng cấu trúc `PDFDocument` gồm trang, block, bảng, thêm bộ lọc text/table, gom block lặp, chuẩn hóa metadata (`loaders/pdf_loader.py`).

2. **Hybrid chunking**  
   - `HybridChunker` kết hợp rule-based, semantic và fixed-size để tạo `ChunkSet` (`pipeline/rag_pipeline.py:302`, `chunkers/hybrid_chunker.py`).  
   - Chunk chứa provenance: trang, block, loại nội dung, table payload khi có.

3. **Chỉ mục BM25 song song**  
   - `BM25IngestManager` nhận `ChunkSet`, rút trích keyword qua spaCy và upsert vào Whoosh (`pipeline/rag_pipeline.py:318`, `BM25/ingest_manager.py`).  
   - Tự động dùng cache `cache/bm25_chunk_cache.json` để bỏ qua chunk không đổi; fallback nếu không có BM25.

4. **Sinh embedding & ghi FAISS**  
   - Embedder mặc định là Ollama Gemma, có thể switch sang BGE-M3 (`pipeline/rag_pipeline.py:78`).  
   - Mọi chunk được embed, lưu metadata phong phú (nguồn, token, provenance) (`pipeline/rag_pipeline.py:332`).  
   - `VectorStore` chuẩn hóa vector (cosine) và ghi cặp `*_vectors_*.faiss` + `*_metadata_map_*.pkl` (`pipeline/vector_store.py`).

5. **Tổng hợp output phụ trợ**  
   - Tạo file log chunk (`data/chunks`), embeddings (`data/embeddings`), FAISS, metadata map, và summary JSON (`pipeline/summary_generator.py`).  
   - Batch ingest cập nhật `batch_summary_*.json`.

## 2. Luồng truy vấn (online)

1. **Điểm vào**  
   - `fetch_retrieval` nhận `query_text`, cấu hình embedder/reranker và token (`pipeline/backend_connector.py:503`).  
   - Cho phép bật/tắt query enhancement, rerank.

2. **Viết lại truy vấn (tùy chọn)**  
   - Tải cấu hình UI (`llm/config_loader.get_config`) & `qem_config.yaml`.  
   - `QueryEnhancementModule.enhance` tạo prompt và gọi LLM để sinh biến thể (`pipeline/query_enhancement/qem_core.py`).  
   - Danh sách kết quả được lọc trắng, dedupe, giới hạn, ghi log JSON.

3. **Chuẩn bị embedder & truy vấn lai**  
   - Tùy chuỗi `embedder_type`, pipeline khởi tạo Ollama hoặc HuggingFace (local/API).  
   - `_fuse_query_embeddings` embed từng biến thể, lấy trung bình làm vector hợp nhất; song song ghép văn bản để chạy BM25 (`pipeline/backend_connector.py:568`).  
   - `RAGRetrievalService.retrieve_hybrid` tải tất cả cặp index hiện có và chạy FAISS + BM25 (`pipeline/backend_connector.py:411`).

4. **Hợp nhất điểm FAISS + BM25**  
   - Chuẩn hóa điểm bằng z-score, trọng số 40/60 mặc định, hợp nhất theo `chunk_id`, giữ metadata gốc (`pipeline/backend_connector.py:283`).  
   - Trả về danh sách kết quả đã sắp xếp theo điểm lai.

5. **Rerank (tùy chọn)**  
   - Khi cấu hình `reranker_type != "none"`, truy vấn Top-N (mặc định gấp 5 lần) và gửi text vào reranker (`pipeline/backend_connector.py:574`).  
   - `RerankerFactory` map string -> enum -> triển khai cụ thể (local/API) (`reranking/reranker_factory.py`).  
   - Kết quả rerank trả index gốc + điểm, được dùng để reorder và bổ sung `rerank_score`.

6. **Đóng gói trả về**  
   - `build_context` cắt ngắn snippet, chèn provenance và bảng nếu có (`pipeline/backend_connector.py:115`).  
   - `to_ui_items` tạo payload hiển thị (title, snippet, vector_similarity, rerank_score, ...).  
   - `fetch_retrieval` trả context, nguồn, truy vấn đã dùng, metadata retrival (embedder, reranker, số lượng, trạng thái lỗi).

## 3. Phân tích chi tiết: Query Enhancement Module

| Thành phần | Mô tả |
| --- | --- |
| Cấu hình | `load_qem_settings` ghép `DEFAULT_SETTINGS` với `qem_config.yaml` nếu có (`pipeline/query_enhancement/qem_core.py:35`). Cho phép cấu hình số biến thể theo ngôn ngữ, backend chính/phụ, log path, lệnh bổ sung. |
| Prompt | `build_prompt` ép LLM trả JSON array (mỗi biến thể ≤25 từ, giữ intent) với ràng buộc ngôn ngữ (`pipeline/query_enhancement/qem_strategy.py`). |
| Backend | `QEMLLMClient` chọn backend dựa vào cấu hình UI > override > fallback Gemini (`pipeline/query_enhancement/qem_lm_client.py:35`). Hỗ trợ `call_gemini` và `call_lmstudio` với override tham số (temperature, top_p, model). |
| Hậu xử lý | `parse_llm_list` ưu tiên JSON, fallback bullet list; `deduplicate_queries` chuẩn hóa lower + trim; `clip_queries` giới hạn theo `max_total_queries`; `log_activity` append JSON log (`pipeline/query_enhancement/qem_utils.py`). |
| Fallback | Nếu LLM lỗi, ghi log lỗi và trả về truy vấn gốc (đảm bảo không chặn retrieval) (`pipeline/query_enhancement/qem_core.py:121`). |

### Điểm mạnh
- Plug-and-play: bật/tắt nhanh qua config, dễ adapt backend.
- Kiểm soát chất lượng: ràng buộc output JSON giúp parser ổn định; dedupe giảm noise.
- Ghi log JSON tuyến tính giúp audit và fine-tune prompt.

### Lưu ý / Cải thiện tiềm năng
- Chưa có cấp phép rate-limit/throttling; cần đảm bảo backend chịu tải.
- `languages` hiện chỉ map `vi`/`en`; có thể cần mapping thêm alias (vd. `vn`).  
- Chưa cache kết quả theo truy vấn → cân nhắc cache cho truy vấn phổ biến.

## 4. Phân tích chi tiết: Rerank layer

| Thành phần | Mô tả |
| --- | --- |
| Interface | `IReranker` quy định `rerank`, `test_connection` và metadata `RerankerProfile` (`reranking/i_reranker.py`). |
| Factory | `RerankerFactory.create` map enum -> triển khai; phân biệt local (Ollama BGE, HF local, MiniLM) và API (HF, Cohere, Jina) (`reranking/reranker_factory.py`). Thiếu token => raise lỗi sớm. |
| Local providers | Dựa trên `BaseLocalReranker`, load model/tokenizer, dùng cross-encoder để tính điểm cho mỗi cặp query-document. Ví dụ MiniLM dùng model tải sẵn ở `rerank_model/model` (`reranking/providers/msmarco_minilm_local_reranker.py`). |
| API providers | Kế thừa `BaseAPIReranker` để chuẩn hóa gọi API, parse response. HF dùng `sentence-transformers/all-MiniLM-L6-v2` làm reranker tương tích logistic (`reranking/providers/bge_m3_hf_api_reranker.py`). |
| Integration | `fetch_retrieval` gọi reranker với danh sách text song song với metadata hybrid. Top-K rerank được chuyển thành index gốc để reorder, giữ nguyên trường `similarity_score` ban đầu và thêm `rerank_score` (`pipeline/backend_connector.py:640`). |
| Thất bại | Bọc try/except → nếu rerank lỗi, log cảnh báo và fallback top-K hybrid đã có (không chặn workflow). |

### Mẹo vận hành
- Đối với local model lớn (BGE), nên chạy trên GPU để tránh throughput thấp.
- Với API, cần truyền `api_tokens` đúng key (`hf`, `cohere`, `jina`) từ layer gọi `fetch_retrieval`.
- `retrieval_top_k = top_k * 5` đảm bảo reranker đủ tập ứng viên; có thể tinh chỉnh theo latency.


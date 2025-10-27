# Query Enhancement Module (QEM)

Thư mục `pipeline/query_enhancement` chứa toàn bộ thành phần phục vụ việc mở rộng truy vấn trước khi thực hiện bước tìm kiếm trong RAG. README này mô tả cấu trúc tệp, vai trò của từng mô-đun và cách chúng phối hợp với các phần khác của hệ thống.

## 1. Tổng quan luồng xử lý
- UI hoặc dịch vụ backend gọi `backend_connector.fetch_retrieval`.
- Hàm này tải cấu hình LLM (qua `get_config`) và khởi tạo `QueryEnhancementModule`.
- `QueryEnhancementModule.enhance` dựng prompt (`qem_strategy.build_prompt`), gọi LLM thông qua `QEMLLMClient`, rồi hậu xử lý kết quả bằng các tiện ích trong `qem_utils`.
- Danh sách truy vấn đã mở rộng được hợp nhất với truy vấn gốc. Module ghi log hoạt động vào `data/logs/qem_activity.jsonl`.
- Bộ truy vấn cuối cùng được sử dụng để:
  1. Tạo embedding trung bình làm đầu vào cho tìm kiếm vector FAISS.
  2. Ghép chuỗi văn bản làm truy vấn BM25.

## 2. Cấu trúc tệp
- `__init__.py`  
  Re-export hai thực thể chính `QueryEnhancementModule` và `load_qem_settings`.
- `qem_core.py`  
  Chứa lớp điều phối trung tâm `QueryEnhancementModule` và logic tải cấu hình.
- `qem_lm_client.py`  
  Bao bọc các lời gọi sang Gemini hoặc LM Studio với cơ chế chọn backend linh hoạt.
- `qem_strategy.py`  
  Xây dựng prompt chuẩn hóa cho LLM, bao gồm yêu cầu ngôn ngữ, định dạng và hướng dẫn thêm.
- `qem_utils.py`  
  Bộ hàm tiện ích: chuẩn hóa/deduplicate truy vấn, parse output LLM, cắt số lượng, ghi log JSONL, tạo chuỗi tóm tắt.
- `qem_config.yaml`  
  Tệp cấu hình mặc định của QEM; có thể chỉnh sửa để thay đổi backend, thông số LLM, ngôn ngữ, đường dẫn log.

## 3. Chi tiết từng mô-đun

### 3.1 `QueryEnhancementModule` (`qem_core.py`)
- **Khởi tạo**  
  - Nhận `app_config` (cấu hình toàn hệ thống) và `qem_settings` (có thể nạp từ YAML qua `load_qem_settings`).  
  - Chuẩn hóa yêu cầu ngôn ngữ (`_normalise_language_requirements`).  
  - Tạo `QEMLLMClient` dựa trên cấu hình đã hợp nhất.
- **`enhance(user_query)`**  
  - Xây dựng prompt qua `build_prompt`.  
  - Gọi LLM: `QEMLLMClient.generate_variants`.  
  - Parse danh sách kết quả (`parse_llm_list`), thêm truy vấn gốc, loại trùng (`deduplicate_queries`), giới hạn tổng số (`clip_queries`).  
  - Ghi nhận hoạt động (`_log_queries` → `log_activity`) và trả về danh sách cuối cùng (fallback về truy vấn gốc nếu lỗi).
- **`load_qem_settings`**  
  - Sao chép `DEFAULT_SETTINGS`, đọc `qem_config.yaml` nếu có (dùng PyYAML).  
  - Hợp nhất cấu hình sâu `_deep_merge` để cho phép override từng phần.
- **Ghi log**  
  - `_log_queries` chuẩn bị payload gồm backend, truy vấn gốc, danh sách biến thể, kết quả raw và lỗi (nếu có), sau đó sử dụng `log_activity`.

### 3.2 `QEMLLMClient` (`qem_lm_client.py`)
- **Nhiệm vụ**: tầng adapter quyết định chọn backend và truyền thông số đến hàm gọi LLM chung của hệ thống.
- **Lựa chọn backend** (`_resolve_backend`):
  1. Ưu tiên `qem_config["backend"]` nếu được đặt.
  2. Nếu không, đọc `app_config["ui"]["default_backend"]`.
  3. Fallback cuối cùng: `qem_config["fallback_backend"]` (mặc định `gemini`).
- **Lời gọi Gemini** (`_call_gemini`):
  - Gửi messages với `system_prompt` và prompt người dùng.
  - Cho phép override `model_name`, `temperature`, `max_tokens`.
- **Lời gọi LM Studio** (`_call_lmstudio`):
  - Convert các tham số số học sang kiểu phù hợp.
  - Trả về chuỗi văn bản raw để caller tự parse.

### 3.3 Prompt strategy (`qem_strategy.py`)
- Hàm `build_prompt` nhận truy vấn và bản đồ `{ngôn_ngữ: số_lượng}`.
- Tính tổng số biến thể cần sinh, chuẩn hóa mô tả ngôn ngữ (English/Vietnamese).  
- Ghép hướng dẫn bắt buộc:
  - Không thay đổi ý định.
  - Giữ câu ngắn (≤ 25 từ).
  - Output **phải** là JSON array.
- Cho phép chèn thêm hướng dẫn tự do (`additional_instructions`) từ cấu hình.

### 3.4 Tiện ích (`qem_utils.py`)
- `normalize_query` / `deduplicate_queries`: Chuẩn hóa và loại bỏ trùng lặp nhưng vẫn giữ nguyên casing đầu ra cho hiển thị.
- `parse_llm_list`: Hỗ trợ cả JSON array lẫn danh sách dạng bullet/đánh số.
- `clip_queries`: Cắt danh sách theo `max_total_queries`.
- `log_activity`: Đảm bảo thư mục log tồn tại, append payload dạng JSON line vào `log_path`. Hỗ trợ tiếng Việt hoặc Unicode nhờ `ensure_ascii=False`.
- `summarise_queries`: Tạo chuỗi gọn gàng phục vụ logging (`logger.debug/info` từ core).

### 3.5 Cấu hình (`qem_config.yaml`)
- `enabled`: Bật/tắt QEM ở runtime.
- `languages`: Số biến thể mong muốn cho từng mã ngôn ngữ (ví dụ `vi: 2`, `en: 2`).
- `max_total_queries`: Giới hạn cứng số truy vấn trả về (bao gồm truy vấn gốc).
- `backend` & `fallback_backend`: Điều khiển backend LLM được chọn.
- `llm_overrides`: Tùy biến thông số gọi LLM (temperature, max_tokens, model…).
- `additional_instructions`: Chuỗi hướng dẫn thêm, append vào prompt.
- `log_path`: Đường dẫn log JSONL (`data/logs/qem_activity.jsonl` theo cấu hình mẫu).

## 4. Điểm tích hợp với các phần khác
- `pipeline/backend_connector.py:28` nhập `QueryEnhancementModule` và `load_qem_settings`.  
  - Trong `fetch_retrieval`, QEM được khởi tạo mỗi lần gọi để đảm bảo đọc cấu hình mới nhất.  
  - Kết quả `expanded_queries` được dùng để fusing embedding (`_fuse_query_embeddings`) và ghép câu cho BM25.
- `pipeline/rag_pipeline.py` không dùng trực tiếp QEM nhưng cung cấp `RAGPipeline` cho `backend_connector` khai thác.
- Log QEM (`log_activity`) tạo file JSONL trong `data/logs`. Các hệ thống giám sát có thể đọc file này để theo dõi backend, lỗi, nội dung prompt-output.


Mọi thay đổi nên đi kèm cập nhật README để đảm bảo người bảo trì dễ dàng nắm được luồng hoạt động của QEM.

# Module query_enhancement — Mở rộng truy vấn cho RAG

Phiên bản: chi tiết module query_enhancement cho hệ thống RAG (Retrieval-Augmented Generation).

Mô tả ngắn: thư mục `pipeline/query_enhancement/` chứa các lớp và hàm chịu trách nhiệm mở rộng truy vấn người dùng thành nhiều biến thể để cải thiện độ chính xác tìm kiếm trong RAG pipeline. Module sử dụng LLM để sinh ra các câu hỏi tương đương, sau đó kết hợp embedding và BM25 search.

## Mục tiêu và phạm vi

- Tách trách nhiệm: chỉ xử lý việc mở rộng truy vấn, không tham gia vào retrieval trực tiếp.
- Cung cấp API rõ ràng để backend_connector sử dụng trong quy trình tìm kiếm.
- Hỗ trợ multiple LLM backends (Gemini, LM Studio) với fallback mechanism.
- Ghi log chi tiết để theo dõi hiệu suất và debug.

## Kiến trúc tổng quan

Thư mục `pipeline/query_enhancement/` gồm các phần chính:

- `qem_core.py` — Lớp điều phối chính QueryEnhancementModule và logic tải cấu hình.
- `qem_lm_client.py` — Client wrapper cho các LLM backends (Gemini, LM Studio).
- `qem_strategy.py` — Xây dựng prompt chuẩn hóa cho LLM.
- `qem_utils.py` — Các hàm tiện ích: parse output, deduplicate, logging.
- `qem_config.yaml` — Cấu hình mặc định cho QEM.

Luồng dữ liệu điển hình:

```text
User query
  -> QueryEnhancementModule.enhance()
  -> QEMLLMClient (Gemini/LM Studio)
  -> Parse & deduplicate variants
  -> Return expanded queries
  -> backend_connector: fuse embeddings + BM25 search
```

## Các module chính (chi tiết)

### qem_core.py

- Mục đích: lớp orchestrator chính cho query enhancement.
- Tính năng:
  - `QueryEnhancementModule()` — constructor với app_config và qem_settings.
  - `enhance(user_query)` — sinh variants qua LLM và trả về danh sách queries.
  - `load_qem_settings()` — tải cấu hình từ YAML với deep merge.
  - `is_enabled()` — kiểm tra QEM có được bật hay không.

### qem_lm_client.py

- Mục đích: adapter cho multiple LLM backends.
- Tính năng:
  - `_resolve_backend()` — tự động chọn backend dựa trên config.
  - `_call_gemini()` / `_call_lmstudio()` — gọi LLM tương ứng.
  - Fallback mechanism khi backend chính fail.

### qem_strategy.py

- Mục đích: xây dựng prompt chuẩn hóa cho LLM.
- Tính năng:
  - `build_prompt()` — tạo prompt với ngôn ngữ, format, instructions.

### qem_utils.py

- Mục đích: utility functions cho QEM.
- Tính năng:
  - `parse_llm_list()` — parse output LLM thành list.
  - `deduplicate_queries()` — loại bỏ trùng lặp.
  - `log_activity()` — ghi log JSONL.
  - `clip_queries()` — giới hạn số lượng queries.

### qem_config.yaml

- Mục đích: cấu hình mặc định cho QEM.
- Cấu hình chính: enabled, languages, max_total_queries, backend, llm_overrides.

## Hành vi "Auto-quét" (Auto-scan) và tích hợp với pipeline

Module `query_enhancement/` tích hợp với pipeline RAG chính:

- **Automatic Enhancement**: Pipeline tự động gọi QueryEnhancementModule khi search queries
- **Multi-query Retrieval**: Mở rộng một query thành nhiều variants để cải thiện recall
- **Fallback Option**: Trả về query gốc nếu LLM enhancement fail
- **Logging**: Ghi lại tất cả hoạt động enhancement vào JSONL log

Ví dụ sử dụng trong backend_connector:

```python
from pipeline.query_enhancement import QueryEnhancementModule

# Khởi tạo QEM
qem = QueryEnhancementModule(app_config)

# Mở rộng query
enhanced_queries = qem.enhance("tìm kiếm thông tin về AI")
# Kết quả: ["tìm kiếm thông tin về AI", "search for AI information", "tìm AI", "artificial intelligence search"]
```

## Contract (tóm tắt API / dữ liệu)

- Input cho `QueryEnhancementModule.enhance()`: user_query (str)
- Output: List[str] chứa query gốc + variants (luôn ít nhất 1 query)
- Input cho `load_qem_settings()`: base_dir (Optional[Path])
- Output: Dict cấu hình QEM đã merge với defaults

## Edge cases và cách xử lý

- LLM API fail: fallback về query gốc, ghi log warning
- Config file missing: sử dụng DEFAULT_SETTINGS
- Empty query: trả về query gốc
- Duplicate variants: tự động deduplicate
- Too many variants: clip theo max_total_queries

## Ví dụ sử dụng (Python)

```python
from pipeline.query_enhancement import QueryEnhancementModule

# Khởi tạo với config mặc định
qem = QueryEnhancementModule(app_config={})

# Mở rộng query đơn giản
queries = qem.enhance("machine learning")
print(queries)
# Output: ['machine learning', 'máy học', 'ML algorithms', 'artificial intelligence']

# Với cấu hình tùy chỉnh
custom_settings = {
    "enabled": True,
    "languages": {"vi": 1, "en": 1},
    "max_total_queries": 3,
    "backend": "gemini"
}
qem_custom = QueryEnhancementModule(app_config={}, qem_settings=custom_settings)
```

## Kiểm thử

- Repository có cấu hình pytest. Để chạy test liên quan tới query_enhancement:

```powershell
python -m pytest tests/query_enhancement/ -v
```

## Hướng dẫn đóng góp (contributors)

- Viết comment và docstring bằng tiếng Việt theo convention của repo.
- Tuân theo pattern: single responsibility principle.
- Thêm unit test cho mọi thay đổi logic enhancement.
- Nếu thêm LLM backend mới, update QEMLLMClient và qem_config.yaml.

## Tài liệu tham chiếu và liên kết

- LLM Integration: `llm/LLM_API.py`, `llm/LLM_LOCAL.py` — cung cấp LLM backends.
- Backend Connector: `pipeline/backend_connector.py` — sử dụng QEM trong retrieval.
- Cấu hình toàn cục: `config/app.yaml`.

## Ghi chú triển khai / Assumptions

- README này mô tả API theo conventions được sử dụng trong repository.
- Gemini/LM Studio APIs phải được cấu hình đúng trong app_config.
- QEM hoạt động bất đồng bộ và không block retrieval pipeline.

## Chi tiết kỹ thuật theo file (tham chiếu mã nguồn)

### `pipeline/query_enhancement/qem_core.py` — lớp QueryEnhancementModule

- Lớp chính: `QueryEnhancementModule`.
- Constructor (tham số chính):
  - `app_config: Dict[str, Any]`
  - `qem_settings: Optional[Dict[str, Any]] = None`
  - `logger: Optional[logging.Logger] = None`

- Methods chính:
  - `enhance(user_query: str) -> List[str]` — mở rộng query, luôn trả về ít nhất [user_query].
  - `is_enabled() -> bool` — kiểm tra QEM enabled.
  - `load_qem_settings(base_dir=None) -> Dict[str, Any]` — tải config từ YAML.

- Error handling: fallback về query gốc khi LLM fail, ghi log warning.

### `pipeline/query_enhancement/qem_lm_client.py` — lớp QEMLLMClient

- Lớp chính: `QEMLLMClient`.
- Constructor: `QEMLLMClient(app_config, qem_config, logger)`.

- Methods:
  - `generate_variants(prompt: str) -> str` — gọi LLM và trả về raw output.
  - `_resolve_backend() -> str` — chọn backend (gemini/lmstudio).
  - `_call_gemini(prompt)` / `_call_lmstudio(prompt)` — gọi LLM cụ thể.

### `pipeline/query_enhancement/qem_strategy.py` — function build_prompt

- Function chính: `build_prompt()`.
- Parameters:
  - `user_query: str`
  - `language_requirements: Mapping[str, int]`
  - `additional_instructions: Optional[str] = None`

- Output: prompt string hoàn chỉnh cho LLM.

### `pipeline/query_enhancement/qem_utils.py` — utility functions

- Functions chính:
  - `parse_llm_list(raw_output: str) -> List[str]` — parse LLM output thành list queries.
  - `deduplicate_queries(queries: Sequence[str]) -> List[str]` — loại bỏ duplicate.
  - `log_activity(log_path: Path, payload: dict, logger)` — ghi log JSONL.
  - `clip_queries(queries: Sequence[str], max_size: int) -> List[str]` — giới hạn số lượng.

### `pipeline/query_enhancement/qem_config.yaml` — cấu hình YAML

- Cấu trúc: flat dict với keys như enabled, languages, max_total_queries, backend, llm_overrides.
- Merge behavior: deep merge với DEFAULT_SETTINGS trong qem_core.py.
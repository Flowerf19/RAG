# Query Enhancement Module (QEM)# Query Enhancement Module (QEM)



[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)



Module mở rộng truy vấn (Query Enhancement) cho hệ thống RAG, sử dụng LLM để tạo nhiều biến thể truy vấn đa ngôn ngữ trước khi tìm kiếm. Tích hợp với FAISS vector search và BM25 keyword search để cải thiện độ chính xác retrieval.Module mở rộng truy vấn (Query Enhancement) cho hệ thống RAG, sử dụng LLM để tạo nhiều biến thể truy vấn đa ngôn ngữ trước khi tìm kiếm. Tích hợp với FAISS vector search và BM25 keyword search để cải thiện độ chính xác retrieval.



## ✨ Tính năng chính## ✨ Tính năng chính



- 🔍 **Multi-Language Query Expansion**: Tạo biến thể truy vấn bằng tiếng Việt và tiếng Anh- 🔍 **Multi-Language Query Expansion**: Tạo biến thể truy vấn bằng tiếng Việt và tiếng Anh

- 🤖 **Multi-LLM Backend**: Hỗ trợ Gemini, LM Studio, và các LLM khác- 🤖 **Multi-LLM Backend**: Hỗ trợ Gemini, LM Studio, và các LLM khác

- 📊 **Activity Logging**: Ghi log JSONL chi tiết cho monitoring và debugging- 📊 **Activity Logging**: Ghi log JSONL chi tiết cho monitoring và debugging

- ⚙️ **Flexible Configuration**: YAML config với runtime override- ⚙️ **Flexible Configuration**: YAML config với runtime override

- 🔄 **Graceful Fallback**: Fallback về truy vấn gốc khi LLM lỗi- 🔄 **Graceful Fallback**: Fallback về truy vấn gốc khi LLM lỗi

- 🎯 **Intent Preservation**: Giữ nguyên ý định truy vấn gốc khi mở rộng- 🎯 **Intent Preservation**: Giữ nguyên ý định truy vấn gốc khi mở rộng



## 🚀 Khởi động nhanh## 🚀 Khởi động nhanh



### Yêu cầu### Yêu cầu



- **Python**: >= 3.13- **Python**: >= 3.13

- **LLM Backend**: Gemini API key hoặc LM Studio server- **LLM Backend**: Gemini API key hoặc LM Studio server

- **Dependencies**: PyYAML, requests- **Dependencies**: PyYAML, requests



### Cấu hình cơ bản### Cấu hình cơ bản



```yaml```yaml

# qem_config.yaml# qem_config.yaml

enabled: trueenabled: true

languages:languages:

  vi: 2    # 2 biến thể tiếng Việt  vi: 2    # 2 biến thể tiếng Việt

  en: 2    # 2 biến thể tiếng Anh  en: 2    # 2 biến thể tiếng Anh

max_total_queries: 5max_total_queries: 5

backend: geminibackend: gemini

fallback_backend: geminifallback_backend: gemini

``````



### Sử dụng cơ bản### Sử dụng cơ bản



```python```python

from pipeline.query_enhancement import QueryEnhancementModule, load_qem_settingsfrom pipeline.query_enhancement import QueryEnhancementModule, load_qem_settings



# Load config# Load config

qem_settings = load_qem_settings()qem_settings = load_qem_settings()



# Initialize module# Initialize module

qem = QueryEnhancementModule(app_config={}, qem_settings=qem_settings)qem = QueryEnhancementModule(app_config={}, qem_settings=qem_settings)



# Enhance query# Enhance query

original_query = "quản lý rủi ro trong IT"original_query = "quản lý rủi ro trong IT"

enhanced_queries = qem.enhance(original_query)enhanced_queries = qem.enhance(original_query)

# Returns: ["quản lý rủi ro trong IT", "IT risk management", "quản lý rủi ro CNTT", ...]# Returns: ["quản lý rủi ro trong IT", "IT risk management", "quản lý rủi ro CNTT", ...]

``````



## 📁 Cấu trúc module## 📁 Cấu trúc module



``````

query_enhancement/query_enhancement/

├── __init__.py              # Export chính: QueryEnhancementModule, load_qem_settings├── __init__.py              # Export chính: QueryEnhancementModule, load_qem_settings

├── qem_core.py              # Lớp điều phối trung tâm├── qem_core.py              # Lớp điều phối trung tâm

├── qem_lm_client.py         # LLM client wrapper (Gemini, LM Studio)├── qem_lm_client.py         # LLM client wrapper (Gemini, LM Studio)

├── qem_strategy.py          # Prompt building strategy├── qem_strategy.py          # Prompt building strategy

├── qem_utils.py             # Utility functions (parse, dedup, logging)├── qem_utils.py             # Utility functions (parse, dedup, logging)

├── qem_config.yaml          # Default configuration├── qem_config.yaml          # Default configuration

└── README.md               # Documentation└── README.md               # Documentation

``````



### Data Flow Architecture### Data Flow Architecture



```mermaid```mermaid

graph TDgraph TD

    A[User Query] --> B[QueryEnhancementModule]    A[User Query] --> B[QueryEnhancementModule]

    B --> C[Build Prompt]    B --> C[Build Prompt]

    C --> D[QEMLLMClient]    C --> D[QEMLLMClient]

    D --> E{Backend}    D --> E{Backend}

    E -->|Gemini| F[Gemini API]    E -->|Gemini| F[Gemini API]

    E -->|LM Studio| G[LM Studio Local]    E -->|LM Studio| G[LM Studio Local]

    F --> H[Raw Response]    F --> H[Raw Response]

    G --> H    G --> H

    H --> I[Parse & Process]    H --> I[Parse & Process]

    I --> J[Deduplicate]    I --> J[Deduplicate]

    J --> K[Add Original]    J --> K[Add Original]

    K --> L[Clip to Max]    K --> L[Clip to Max]

    L --> M[Enhanced Queries]    L --> M[Enhanced Queries]



    M --> N[FAISS Embedding Fusion]    M --> N[FAISS Embedding Fusion]

    M --> O[BM25 Query Concat]    M --> O[BM25 Query Concat]



    style A fill:#e1f5fe    style A fill:#e1f5fe

    style M fill:#c8e6c9    style M fill:#c8e6c9

``````



## 🔧 Sử dụng trong code## ⚙️ Cấu hình



### Basic Usage### qem_config.yaml



```python```yaml

from pipeline.query_enhancement import QueryEnhancementModule, load_qem_settings# Enable/disable QEM

enabled: true

# Load settings from YAML

qem_settings = load_qem_settings()# Language variants to generate

languages:

# Initialize with app config  vi: 2        # Vietnamese variants

app_config = {}  # Your app configuration  en: 2        # English variants

qem = QueryEnhancementModule(app_config, qem_settings)

# Maximum total queries (including original)

# Enhance single querymax_total_queries: 5

query = "service management process"

enhanced = qem.enhance(query)# LLM Backend selection

print(f"Original: {query}")backend: gemini                    # Primary backend

print(f"Enhanced: {enhanced}")fallback_backend: gemini          # Fallback if primary fails

```

# LLM parameters override

### Integration với RAG Pipelinellm_overrides:

  model_name: "gemini-1.5-flash"

```python  temperature: 0.3

# Trong backend_connector.py  max_tokens: 200

from pipeline.query_enhancement import QueryEnhancementModule, load_qem_settings

# Additional prompt instructions

def fetch_retrieval(query, ...):additional_instructions: |

    # Load QEM settings  Focus on IT service management terminology.

    qem_settings = load_qem_settings()  Include synonyms for technical terms.



    # Initialize QEM# Logging configuration

    qem = QueryEnhancementModule(app_config, qem_settings)log_path: "data/logs/qem_activity.jsonl"

```

    # Enhance query

    enhanced_queries = qem.enhance(query)### Environment Variables



    # Use enhanced queries for retrieval```bash

    fused_embedding = _fuse_query_embeddings(enhanced_queries)# Gemini API (required for gemini backend)

    bm25_query = " ".join(enhanced_queries)export GOOGLE_API_KEY="your-gemini-api-key"



    # Continue with FAISS + BM25 search...# LM Studio (required for lmstudio backend)

```export LM_STUDIO_BASE_URL="http://localhost:1234"

```

### Custom Configuration

### 3.2 `QEMLLMClient` (`qem_lm_client.py`)

```python- **Nhiệm vụ**: tầng adapter quyết định chọn backend và truyền thông số đến hàm gọi LLM chung của hệ thống.

# Override settings programmatically- **Lựa chọn backend** (`_resolve_backend`):

custom_settings = {  1. Ưu tiên `qem_config["backend"]` nếu được đặt.

    "enabled": True,  2. Nếu không, đọc `app_config["ui"]["default_backend"]`.

    "languages": {"vi": 3, "en": 1},  3. Fallback cuối cùng: `qem_config["fallback_backend"]` (mặc định `gemini`).

    "max_total_queries": 5,- **Lời gọi Gemini** (`_call_gemini`):

    "backend": "lmstudio",  - Gửi messages với `system_prompt` và prompt người dùng.

    "llm_overrides": {  - Cho phép override `model_name`, `temperature`, `max_tokens`.

        "temperature": 0.3,- **Lời gọi LM Studio** (`_call_lmstudio`):

        "max_tokens": 200  - Convert các tham số số học sang kiểu phù hợp.

    }  - Trả về chuỗi văn bản raw để caller tự parse.

}

### 3.3 Prompt strategy (`qem_strategy.py`)

qem = QueryEnhancementModule(app_config, custom_settings)- Hàm `build_prompt` nhận truy vấn và bản đồ `{ngôn_ngữ: số_lượng}`.

```- Tính tổng số biến thể cần sinh, chuẩn hóa mô tả ngôn ngữ (English/Vietnamese).  

- Ghép hướng dẫn bắt buộc:

## ⚙️ Cấu hình  - Không thay đổi ý định.

  - Giữ câu ngắn (≤ 25 từ).

### qem_config.yaml  - Output **phải** là JSON array.

- Cho phép chèn thêm hướng dẫn tự do (`additional_instructions`) từ cấu hình.

```yaml

# Enable/disable QEM### 3.4 Tiện ích (`qem_utils.py`)

enabled: true- `normalize_query` / `deduplicate_queries`: Chuẩn hóa và loại bỏ trùng lặp nhưng vẫn giữ nguyên casing đầu ra cho hiển thị.

- `parse_llm_list`: Hỗ trợ cả JSON array lẫn danh sách dạng bullet/đánh số.

# Language variants to generate- `clip_queries`: Cắt danh sách theo `max_total_queries`.

languages:- `log_activity`: Đảm bảo thư mục log tồn tại, append payload dạng JSON line vào `log_path`. Hỗ trợ tiếng Việt hoặc Unicode nhờ `ensure_ascii=False`.

  vi: 2        # Vietnamese variants- `summarise_queries`: Tạo chuỗi gọn gàng phục vụ logging (`logger.debug/info` từ core).

  en: 2        # English variants

### 3.5 Cấu hình (`qem_config.yaml`)

# Maximum total queries (including original)- `enabled`: Bật/tắt QEM ở runtime.

max_total_queries: 5- `languages`: Số biến thể mong muốn cho từng mã ngôn ngữ (ví dụ `vi: 2`, `en: 2`).

- `max_total_queries`: Giới hạn cứng số truy vấn trả về (bao gồm truy vấn gốc).

# LLM Backend selection- `backend` & `fallback_backend`: Điều khiển backend LLM được chọn.

backend: gemini                    # Primary backend- `llm_overrides`: Tùy biến thông số gọi LLM (temperature, max_tokens, model…).

fallback_backend: gemini          # Fallback if primary fails- `additional_instructions`: Chuỗi hướng dẫn thêm, append vào prompt.

- `log_path`: Đường dẫn log JSONL (`data/logs/qem_activity.jsonl` theo cấu hình mẫu).

# LLM parameters override

llm_overrides:## 🚨 Troubleshooting

  model_name: "gemini-1.5-flash"

  temperature: 0.3### Common Issues

  max_tokens: 200

#### LLM Backend Connection Failed

# Additional prompt instructions

additional_instructions: |```python

  Focus on IT service management terminology.# Check backend availability

  Include synonyms for technical terms.from pipeline.query_enhancement.qem_lm_client import QEMLLMClient



# Logging configurationclient = QEMLLMClient(app_config, qem_settings)

log_path: "data/logs/qem_activity.jsonl"try:

```    test_response = client.generate_variants("test query", {"en": 1})

    print("Backend working:", test_response)

### Environment Variablesexcept Exception as e:

    print("Backend error:", e)

```bash```

# Gemini API (required for gemini backend)

export GOOGLE_API_KEY="your-gemini-api-key"#### Invalid YAML Configuration



# LM Studio (required for lmstudio backend)```python

export LM_STUDIO_BASE_URL="http://localhost:1234"# Validate config

```import yaml

from pipeline.query_enhancement import load_qem_settings

## 📊 Monitoring & Logging

try:

### Activity Logs    settings = load_qem_settings()

    print("Config loaded successfully")

QEM ghi log chi tiết vào `data/logs/qem_activity.jsonl`:    print("Languages:", settings.get("languages"))

except yaml.YAMLError as e:

```json    print("YAML error:", e)

{```

  "timestamp": "2025-10-28T10:30:00Z",

  "backend": "gemini",#### No Query Variants Generated

  "original_query": "IT service management",

  "enhanced_queries": ["IT service management", "ITSM", "quản lý dịch vụ CNTT"],```python

  "raw_response": "[\"IT service management\", \"ITSM\", \"quản lý dịch vụ CNTT\"]",# Debug QEM processing

  "error": null,qem = QueryEnhancementModule(app_config, qem_settings)

  "processing_time_ms": 1250

}# Enable debug logging

```import logging

logging.basicConfig(level=logging.DEBUG)

### Log Analysis

result = qem.enhance("test query")

```pythonprint("Result:", result)

import json```



# Read QEM activity logs### Performance Tuning

with open("data/logs/qem_activity.jsonl", "r", encoding="utf-8") as f:

    for line in f:```yaml

        entry = json.loads(line)# High performance config

        print(f"Query: {entry['original_query']}")enabled: true

        print(f"Enhanced: {len(entry['enhanced_queries'])} variants")languages:

        print(f"Backend: {entry['backend']}")  vi: 1

        print("---")  en: 1

```max_total_queries: 3

llm_overrides:

## 🚨 Troubleshooting  temperature: 0.1      # Lower temperature for consistency

  max_tokens: 100       # Shorter responses

### Common Issues```



#### LLM Backend Connection Failed## 🧪 Testing



```python### Unit Tests

# Check backend availability

from pipeline.query_enhancement.qem_lm_client import QEMLLMClient```python

# Test QEM components

client = QEMLLMClient(app_config, qem_settings)from pipeline.query_enhancement.qem_utils import parse_llm_list, deduplicate_queries

try:

    test_response = client.generate_variants("test query", {"en": 1})# Test parsing

    print("Backend working:", test_response)raw_response = '["query1", "query2", "query1"]'

except Exception as e:parsed = parse_llm_list(raw_response)

    print("Backend error:", e)assert parsed == ["query1", "query2", "query1"]

```

# Test deduplication

#### Invalid YAML Configurationdeduped = deduplicate_queries(parsed)

assert deduped == ["query1", "query2"]

```python```

# Validate config

import yaml### Integration Tests

from pipeline.query_enhancement import load_qem_settings

```python

try:# Full QEM pipeline test

    settings = load_qem_settings()from pipeline.query_enhancement import QueryEnhancementModule

    print("Config loaded successfully")

    print("Languages:", settings.get("languages"))qem = QueryEnhancementModule({}, {"enabled": True, "languages": {"en": 1}})

except yaml.YAMLError as e:result = qem.enhance("test query")

    print("YAML error:", e)

```assert len(result) >= 1  # At least original query

assert "test query" in result  # Original preserved

#### No Query Variants Generated```



```python## 🔧 Development

# Debug QEM processing

qem = QueryEnhancementModule(app_config, qem_settings)### Adding New LLM Backend



# Enable debug logging```python

import logging# In qem_lm_client.py

logging.basicConfig(level=logging.DEBUG)def _call_new_backend(self, prompt, **kwargs):

    # Implement new backend logic

result = qem.enhance("test query")    response = call_new_llm_api(prompt, **kwargs)

print("Result:", result)    return response

```

# Update _resolve_backend method

### Performance Tuningdef _resolve_backend(self):

    # Add new backend option

```yaml    if self.config.get("backend") == "new_backend":

# High performance config        return self._call_new_backend

enabled: true```

languages:

  vi: 1### Custom Prompt Strategy

  en: 1

max_total_queries: 3```python

llm_overrides:# In qem_strategy.py

  temperature: 0.1      # Lower temperature for consistencydef build_custom_prompt(query, language_map, additional_instructions=""):

  max_tokens: 100       # Shorter responses    # Custom prompt building logic

```    prompt = f"Generate variants for: {query}\n"

    prompt += f"Languages: {language_map}\n"

## 🧪 Testing    if additional_instructions:

        prompt += f"Additional: {additional_instructions}\n"

### Unit Tests    return prompt

```

```python

# Test QEM components## 📈 Performance Metrics

from pipeline.query_enhancement.qem_utils import parse_llm_list, deduplicate_queries

- **Response Time**: 500-2000ms per query (depends on LLM backend)

# Test parsing- **Success Rate**: >95% với Gemini, >90% với LM Studio

raw_response = '["query1", "query2", "query1"]'- **Query Expansion**: 2-5x số lượng truy vấn

parsed = parse_llm_list(raw_response)- **Memory Usage**: <50MB cho module

assert parsed == ["query1", "query2", "query1"]

## 🤝 Contributing

# Test deduplication

deduped = deduplicate_queries(parsed)### Code Standards

assert deduped == ["query1", "query2"]

```- **Language**: Vietnamese comments, English docstrings

- **Style**: Black formatter, isort imports

### Integration Tests- **Testing**: pytest với coverage > 80%

- **Documentation**: Update README cho breaking changes

```python

# Full QEM pipeline test### Architecture Guidelines

from pipeline.query_enhancement import QueryEnhancementModule

- **Single Responsibility**: Mỗi module một nhiệm vụ rõ ràng

qem = QueryEnhancementModule({}, {"enabled": True, "languages": {"en": 1}})- **Error Handling**: Graceful degradation, detailed logging

result = qem.enhance("test query")- **Configuration**: YAML-first với programmatic override

- **Testing**: Unit tests cho utilities, integration tests cho pipeline

assert len(result) >= 1  # At least original query

assert "test query" in result  # Original preserved## 📞 Support

```

- **Issues**: [GitHub Issues](https://github.com/Flowerf19/RAG/issues)

## 🔧 Development- **Discussions**: [GitHub Discussions](https://github.com/Flowerf19/RAG/discussions)

- **Documentation**: See module READMEs for technical details

### Adding New LLM Backend

---

```python

# In qem_lm_client.py*Module này là phần của hệ thống RAG Pipeline. Xem README chính để biết thêm về kiến trúc tổng thể.*

def _call_new_backend(self, prompt, **kwargs):
    # Implement new backend logic
    response = call_new_llm_api(prompt, **kwargs)
    return response

# Update _resolve_backend method
def _resolve_backend(self):
    # Add new backend option
    if self.config.get("backend") == "new_backend":
        return self._call_new_backend
```

### Custom Prompt Strategy

```python
# In qem_strategy.py
def build_custom_prompt(query, language_map, additional_instructions=""):
    # Custom prompt building logic
    prompt = f"Generate variants for: {query}\n"
    prompt += f"Languages: {language_map}\n"
    if additional_instructions:
        prompt += f"Additional: {additional_instructions}\n"
    return prompt
```

## 📈 Performance Metrics

- **Response Time**: 500-2000ms per query (depends on LLM backend)
- **Success Rate**: >95% với Gemini, >90% với LM Studio
- **Query Expansion**: 2-5x số lượng truy vấn
- **Memory Usage**: <50MB cho module

## 🤝 Contributing

### Code Standards

- **Language**: Vietnamese comments, English docstrings
- **Style**: Black formatter, isort imports
- **Testing**: pytest với coverage > 80%
- **Documentation**: Update README cho breaking changes

### Architecture Guidelines

- **Single Responsibility**: Mỗi module một nhiệm vụ rõ ràng
- **Error Handling**: Graceful degradation, detailed logging
- **Configuration**: YAML-first với programmatic override
- **Testing**: Unit tests cho utilities, integration tests cho pipeline

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Flowerf19/RAG/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Flowerf19/RAG/discussions)
- **Documentation**: See module READMEs for technical details

---

*Module này là phần của hệ thống RAG Pipeline. Xem README chính để biết thêm về kiến trúc tổng thể.*